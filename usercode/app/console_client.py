import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.helper import panel_print

import socketio
import socketio.exceptions
import asyncio
import asyncio.exceptions
import argparse
from collections import defaultdict

from application_classes import Arguments

from rich.pretty import Pretty
from rich.console import Console
from rich.live import Live
from rich.rule import Rule
from rich.tree import Tree

console = Console()
sio = socketio.AsyncClient()

#===============================================================================================================

data_to_send = None
full_text = ""
receiving_fragments = False
live: Live = None
times = {}
end: bool = False
args = None

#===============================================================================================================

async def reset_state():
    global full_text
    global receiving_fragments
    global live
    global times
    global end

    full_text = ""
    receiving_fragments = False
    live = None
    times = {}
    end = False

#===============================================================================================================

async def print_times():
    #Print times
    #------------------------------------------------------------------------------
    mytimes = defaultdict(float, {k:round(v, 3) for k,v in times.items()})

    tree = Tree(f"[green]Total time: [cyan]{sum(mytimes.values()):.3f}s[/cyan]")
    tree.add(f"[green]Elasticsearch time: [cyan]{mytimes['elastic']:.3f}s[/cyan]")
    tree.add(f"[green]Query encoding: [cyan]{mytimes['query_encode']:.3f}s[/cyan]")
    tree.add(f"[green]Cluster retrieval: [cyan]{mytimes['cluster_retrieval']:.3f}s[/cyan]")

    score_tree = tree.add(f"[green]Cross-scores: [cyan]{sum(v for k,v in mytimes.items() if k.startswith('cross_score')):.3f}s[/cyan]")
    for k,v in mytimes.items():
        if k.startswith('cross_score'):
            score_tree.add(f"[green]Cluster {k[12:]}: [cyan]{v:.3f}s[/cyan]")

    context_tree = tree.add(f"[green]Context expansion: [cyan]{sum(v for k,v in mytimes.items() if k.startswith('context_expansion')):.3f}s[/cyan]")
    for k,v in mytimes.items():
        if k.startswith('context_expansion'):
            context_tree.add(f"[green]Cluster {k[18:]}: [cyan]{v:.3f}s[/cyan]")

    summary_tree = tree.add(f"[green]Summarization[/green]: [cyan]{mytimes['summary_time']}s[/cyan]")
    summary_tree.add(f"[green]Response time[/green]: [cyan]{mytimes['summary_response_time']:.3f}s[/cyan]")

    console.print(tree)
    print()

#===============================================================================================================

@sio.event(namespace="/query")
async def connect():
    console.print("[green]Connected to server successfully[/green]\n")
    #Send query and arguments to the server
    await sio.emit('init_query', data_to_send, namespace="/query")

@sio.event(namespace="/query")
async def disconnect():
    global end
    console.print("Disconnecting from server...\n")
    end = True

@sio.event(namespace="/query")
async def message(data):
    console.print(f"[cyan][MESSAGE][/cyan]: {data}")

#===============================================================================================================

@sio.on("time", namespace="/query")
async def ev_time(data):
    global times
    times = {**times, **data}

#---

@sio.on("ansi_text", namespace="/query")
async def ev_ansi_text(data):
    print(data, end="")

#---

@sio.on("info", namespace="/query")
async def ev_info(data):
    console.print(f"[green]INFO:[/green] {data}\n")

#---

@sio.on("cosine_sim", namespace="/query")
async def ev_cosine_sim(data):
    panel_print([f"[green]{i:02}.[/green] Cluster [green]{cluster['id']}[/green] with score [cyan]{cluster['sim']:.3f}[/cyan]" for i, cluster in enumerate(data)], title="Retrieved clusters based on cosine similarity")

#---

@sio.on("cluster_stats", namespace="/query")
async def ev_cluster_stats(data):
    panel_print([Pretty(stats) for stats in data], title="Cluster Stats")

#---

@sio.on("cross_scores", namespace="/query")
async def ev_cross_scores(data):
    panel_print([f"Cluster [green]{cluster['id']}[/green] score: [cyan]{cluster['cross_score']:.3f}[/cyan]" for cluster in data], title="Cross-encoder scores of the selected clusters")

#---

@sio.on("context_expansion_progress", namespace="/query")
async def ev_context_expansion_progress(data):
    if args.print:
        console.print(Rule(title=f"Cluster {data}", align="center"))

#---

@sio.on("cross_scores_2", namespace="/query")
async def ev_cross_scores_2(data):
    panel_print([f"Cluster [green]{cluster['id']}[/green] score: [cyan]{cluster['original_score']}[/cyan] -> [cyan]{cluster['new_score']:.3f}[/cyan] ([green]+{round(cluster['new_score'] - cluster['original_score'], 3):.3f}[/green])" for cluster in data], title="Cross-encoder scores of the selected clusters after context expansion")
    panel_print([f"Cluster [green]{cluster['id']}[/green] score: [cyan]{cluster['selected_score']:.3f}[/cyan]" for cluster in data], title="Cross-encoder scores (only selected candidates considered)")
    
#---

@sio.on("fragment", namespace="/query")
async def ev_fragment(data):
    global full_text
    global receiving_fragments
    global live
    if not receiving_fragments:
        live = Live(panel_print(return_panel=True), refresh_per_second=10)
        live.start()
    receiving_fragments = True
    full_text += data

#---

@sio.on("fragment_with_citation", namespace="/query")
async def ev_fragment_with_citation(data):
    global full_text
    data = data
    cite = data['citation']
    temp = data['fragment']
    temp = temp.replace("<citation>", f"[cyan]<{cite['doc']}_{cite['start']}-{cite['end']}>[/cyan]")
    full_text += temp

#---
    
@sio.on("end", namespace="/query")
async def ev_end(data):
    global end
    global live
    if live is not None:
        live.stop()
    end = True

#---

@sio.on("error", namespace="/query")
async def ev_error(data):
    global live
    if live is not None:
        live.stop()
    console.print(f"[red]{data}[/red]\n")
    raise Exception
            

#===============================================================================================================

async def main_loop():
    while True:
        if receiving_fragments:
            live.update(panel_print(full_text, return_panel=True))
        if end:
            await print_times()
            print("Stopping...")
            return
        await asyncio.sleep(0.001)

#===============================================================================================================

async def main():
    global data_to_send
    global args

    parser = argparse.ArgumentParser()
    Arguments.setup_arguments(parser)
    args = Arguments.from_argparse(parser.parse_args())

    query = "What are the primary behaviours and lifestyle factors that contribute to childhood obesity"
    experiment_list = args.x.split(",")
    
    #For each specified experiment, send a request to the server
    #At the end, we will compare the results
    for experiment in experiment_list:
        args.x = experiment
        try:
            data_to_send = {
                'query': query,
                'args': args.to_dict(ignore_defaults=True, ignore_client_args=True)
            }

            await sio.connect("http://localhost:1225", namespaces=["/query"])
            sio.start_background_task(main_loop)
            await sio.wait()

        except (KeyboardInterrupt, asyncio.exceptions.CancelledError):
            await sio.disconnect()
        except socketio.exceptions.ConnectionError:
            console.print("[red]Failed to connect to server[/red]")

#===============================================================================================================

if __name__ == "__main__":
    asyncio.run(main())
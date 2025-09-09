import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.helper import panel_print
from mypackage.experiments import ExperimentManager

import socketio
import socketio.exceptions
import asyncio
import asyncio.exceptions
import argparse
from collections import defaultdict

from application_helper import Arguments, create_time_tree

from rich.pretty import Pretty
from rich.console import Console
from rich.live import Live
from rich.rule import Rule

import warnings
import shutil

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

@sio.event(namespace="/query")
async def connect():
    console.print("[green]Connected to server successfully[/green]\n")
    console.print(f"[white]Query: [green]\"{data_to_send['query']['text']}\"[/green]\n")
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
    console.print(f"[green]INFO:[/green] {data}")

#---

@sio.on("warn", namespace="/query")
async def ev_warn(data):
    console.print(f"[yellow]WARNING:[/yellow] {data}")

#---

@sio.on("docs", namespace="/query")
async def ev_docs(data):
    console.print(f"Returned docs: {data}\n")

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
    data = data #wtf is this
    cite = data['citation']
    temp = data['fragment']
    temp = temp.replace("<citation>", f"[cyan]<{cite['doc']}_{cite['start']}-{cite['end']}>[/cyan]")
    full_text += temp

#---

@sio.on("summary_end", namespace="/query")
async def ev_summary_end():
    global live, full_text, receiving_fragments
    if live is not None:
        live.stop()
        live = None
    receiving_fragments = False
    full_text = ""

#---
    
@sio.on("end", namespace="/query")
async def ev_end(data):
    global end, live
    if data['status'] < 0:
        console.print(f"[red]{data['msg']}[/red]")
    if live is not None:
        live.stop()
    await sio.disconnect()

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
            tree,_ = create_time_tree(times)
            console.print(tree)
            print()
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

    #Determine query
    queries = ExperimentManager("../common/experiments.json").get_queries(args.query, args.index)
    if len(queries) > 1:
        warnings.warn("Specified more than one query. Only the first one will be used.")
    query = queries[0]

    experiment_list = args.x.split(",")
    
    #For each specified experiment, send a request to the server
    #At the end, we will compare the results
    for experiment in experiment_list:
        args.x = experiment
        try:
            status, msg = args.validate()

            if status:
                data_to_send = {
                    'query': query.data(),
                    'args': args.to_dict(ignore_defaults=True, ignore_client_args=True),
                    'console_width': shutil.get_terminal_size().columns,
                    'store_as': args.store_as
                }

                #Connect and send data
                await sio.connect("http://localhost:1225", namespaces=["/query"])
                sio.start_background_task(main_loop)
                await sio.wait()
            else:
                console.print(f"[red]{msg}[/red]")

        except (KeyboardInterrupt, asyncio.exceptions.CancelledError):
            await sio.disconnect()
        except socketio.exceptions.ConnectionError:
            console.print("[red]Failed to connect to server[/red]")

#===============================================================================================================

if __name__ == "__main__":
    asyncio.run(main())
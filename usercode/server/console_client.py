import os
import sys
sys.path.append(os.path.abspath("../.."))

from typing import Literal, get_origin, get_args, Any
from dataclasses import fields
import argparse
import sseclient
import json

from mypackage.helper import panel_print

from rich.pretty import Pretty
from rich.console import Console
from rich.live import Live
from rich.rule import Rule
from rich.padding import Padding
from rich.tree import Tree

from classes import Message, Arguments
from collections import defaultdict

console = Console()

#===============================================================================================================

full_text = ""
receiving_fragments = False
live: Live = None
times = {}
end: bool = False

#===============================================================================================================

def reset_state():
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

def print_times():
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

def message_handler(msg: Message):
    global receiving_fragments
    global live
    global full_text
    global times
    global end

    if msg is None:
        return

    match msg.type:
        case 'time':
            times = {**times, **msg.contents}
        case 'ansi_text':
            print(msg.contents, end="")
        case 'info':
            console.print(f"[green]INFO:[/green] {msg.contents}\n")
        case 'cosine_sim':
            panel_print([f"[green]{i:02}.[/green] Cluster [green]{cluster['id']}[/green] with score [cyan]{cluster['sim']:.3f}[/cyan]" for i, cluster in enumerate(msg.contents)], title="Retrieved clusters based on cosine similarity")
        case 'cluster_stats':
            panel_print([Pretty(stats) for stats in msg.contents], title="Cluster Stats")
        case 'cross_scores':
            panel_print([f"Cluster [green]{cluster['id']}[/green] score: [cyan]{cluster['cross_score']:.3f}[/cyan]" for cluster in msg.contents], title="Cross-encoder scores of the selected clusters")
        case 'context_expansion_progress':
            if args.print:
                console.print(Rule(title=f"Cluster {msg.contents}", align="center"))
        case 'cross_scores_2':
            panel_print([f"Cluster [green]{cluster['id']}[/green] score: [cyan]{cluster['original_score']}[/cyan] -> [cyan]{cluster['new_score']:.3f}[/cyan] ([green]+{round(cluster['new_score'] - cluster['original_score'], 3):.3f}[/green])" for cluster in msg.contents], title="Cross-encoder scores of the selected clusters after context expansion")
            panel_print([f"Cluster [green]{cluster['id']}[/green] score: [cyan]{cluster['selected_score']:.3f}[/cyan]" for cluster in msg.contents], title="Cross-encoder scores (only selected candidates considered)")
        case 'fragment':
            if not receiving_fragments:
                live = Live(panel_print(return_panel=True), refresh_per_second=10)
                live.start()
            receiving_fragments = True
            full_text += msg.contents
        case 'fragment_with_citation':
            data = msg.contents
            cite = data['citation']
            temp = data['fragment']
            temp = temp.replace("<citation>", f"[cyan]<{cite['doc']}_{cite['start']}-{cite['end']}>[/cyan]")
            full_text += temp
        case 'end':
            if live is not None:
                live.stop()
            end = True
        case 'error':
            if live is not None:
                live.stop()
            console.print(f"[red]{msg.contents}[/red]\n")
            raise Exception

#===============================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Arguments.setup_arguments(parser)
    args = Arguments.from_argparse(parser.parse_args())

    query = "What are the primary behaviours and lifestyle factors that contribute to childhood obesity"
    experiment_list = args.x.split(",")

    #For each specified experiment, send a request to the server
    #At the end, we will compare the results
    for experiment in experiment_list:
        try:
            messages = sseclient.SSEClient('http://localhost:4625/query', params={
                'q': query,
                **args.get_dict(ignore_defaults=True),
                'x': experiment
            })

            #Receiving messages
            #---------------------------------------------------------------
            message_stream = map(lambda event: Message.from_sse_event(event), messages)

            console.print(f"\n[green]Query:[/green] [#ffffff]\"{query}\"[/#ffffff]\n")
            
            for msg in message_stream:
                message_handler(msg)

                #What to do with the new state
                if receiving_fragments:
                    live.update(panel_print(full_text, return_panel=True))
                if end:
                    messages.resp.close()
                    print_times()
                    break

            #Prepare for next experiment
            if len(experiment_list) > 1:
                reset_state()

        except (Exception, KeyboardInterrupt):
            messages.resp.close()
            print_times()
            print("Stopping...")
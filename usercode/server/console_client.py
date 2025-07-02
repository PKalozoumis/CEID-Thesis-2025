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

console = Console()

#===============================================================================================================

if __name__ == "__main__":

    #Arguments
    #------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    for f in fields(Arguments):
        if f.type is bool:
            parser.add_argument(f"--{f.name}", action="store_true", dest=f.name, default=f.default, help=f.metadata['help'])
            if f.default == True:
                parser.add_argument(f"--no-{f.name}", action="store_false", dest=f.name, help=f.metadata['help'])
        elif get_origin(f.type) is Literal:
            parser.add_argument(f"-{f.name}", action="store", type=str, default=f.default, help=f.metadata['help'], choices=list(get_args(f.type)))
        else:
            parser.add_argument(f"-{f.name}", action="store", type=f.type, default=f.default, help=f.metadata['help'])

    args = parser.parse_args()

    #Connect to the server
    #------------------------------------------------------------------------------------
    try:
        messages = sseclient.SSEClient('http://localhost:4625/query', params={'q': "My name is Edwin. I made the Mimic", 'console_messages': 1})

        full_text = ""
        receiving_fragments = False
        live: Live = None
        stop: bool = False

        #Message handler
        #---------------------------------------------------------------
        def message_handler(msg: Message):
            global receiving_fragments
            global live
            global full_text

            match msg.type:
                case 'query':
                    console.print(f"\n[green]Query:[/green] {msg.contents}\n")
                case 'cosine_sim':
                    panel_print([f"[green]{i:02}.[/green] Cluster [green]{cluster['id']}[/green] with score [cyan]{cluster['sim']:.3f}[/cyan]" for i, cluster in enumerate(msg.contents)], title="Retrieved clusters based on cosine similarity")
                case 'fragment':
                    if not receiving_fragments:
                        live = Live(panel_print(return_panel=True), refresh_per_second=10)
                        live.start()
                    
                    receiving_fragments = True
                    full_text += msg.contents
                case 'end':
                    live.stop()
                    stop = True


        #Receiving messages
        #---------------------------------------------------------------
        message_stream = map(lambda event: Message.from_sse_event(event), messages)
        

        for msg in message_stream:
            message_handler(msg)

            #What to do with the new state
            if receiving_fragments:
                live.update(panel_print(full_text, return_panel=True))
            if stop:
                break

        console.log("Done")

    except KeyboardInterrupt:
        sys.stdout.write('\b\b')
        console.print("Stopping...")
import os
import sys
sys.path.append(os.path.abspath("../.."))

from flask import Flask, request, make_response, Response
from flask_socketio import SocketIO, send
from usercode.server.application_pipeline import pipeline
from rich.console import Console
from classes import Arguments
from mypackage.llm import LLMSession
import argparse

console = Console()
app = Flask(__name__)
socketio = SocketIO(app)

#==================================================================================

cached_prompt = False
stop_dict = {'force_stop': False, 'stopped': False}
server_args = None

@socketio.on("connect", namespace="/query")
def connect():
    console.print(f"[green]Client {request.sid} connected[/green]")
    stop_dict['force_stop'] = False
    stop_dict['stopped'] = False

@socketio.on("init_query", namespace="/query")
def query(data):
    query = data['query']
    args = Arguments(**data['args'])
    console.print(args)
    pipeline(query, stop_dict, socket=socketio, args=args, server_args=server_args)

@socketio.on('disconnect', namespace="/query")
def on_disconnect():
    stop_dict['force_stop'] = True
    console.print(f"[red]Client {request.sid} disconnected[/red]")

#Main
#Only if running from pure flask
#Ideally you want to run through some other WSGI server
#==================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--llm-backend", action="store", type=str, choices=["llamacpp", "lmstudio"], default="lmstudio")
    parser.add_argument("--host", action="store", type=str, choices=["llamacpp", "lmstudio"], default="localhost:8080")
    parser.add_argument("-p", "--port", action="store", type=int, help="Server port", default=1225)
    server_args = parser.parse_args()

    #LLMSession.create(args.llm_backend).cache_system_prompt()
    socketio.run(app, port=server_args.port)
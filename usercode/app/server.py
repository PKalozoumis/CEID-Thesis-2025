import os
import sys
sys.path.append(os.path.abspath("../.."))

import argparse

#==================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--llm-backend", action="store", type=str, choices=["llamacpp", "lmstudio"], default="lmstudio")
    parser.add_argument("--host", action="store", type=str, default="localhost:1234")
    parser.add_argument("-p", "--port", action="store", type=int, help="Server port", default=1225)
    parser.add_argument("--no-cache", action="store_true", help="Disable system prompt caching", default=False)
    parser.add_argument("-db", action="store", type=str, default='mongo', help="Database to store the preprocessing results in", choices=['mongo', 'pickle'])
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    server_args = parser.parse_args()

#==================================================================================

from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from rich.console import Console
from mypackage.llm import LLMSession

from application_pipeline import pipeline
from application_classes import Arguments

console = Console()
app = Flask(__name__)
socketio = SocketIO(app)

#==================================================================================

cached_prompt = False
stop_dict = {'force_stop': False, 'stopped': False, 'conn': None}

@socketio.on("connect", namespace="/query")
def connect():
    console.print(f"[green]Client {request.sid} connected[/green]")
    stop_dict['force_stop'] = False
    stop_dict['stopped'] = False

@socketio.on("init_query", namespace="/query")
def query(data):
    try:
        query = data['query']
        args = Arguments(**data['args'])
        console.print(args)
        pipeline(query, stop_dict, socket=socketio, args=args, server_args=server_args, console_width=data.get('console_width', None))
    except:
        socketio.emit('end', {'status': -1, 'msg': "Internal server error"}, namespace='/query')
        raise

@socketio.on('disconnect', namespace="/query")
def on_disconnect():
    if stop_dict['conn'] is not None:
        stop_dict['conn'].disconnect()
    stop_dict['force_stop'] = True
    console.print(f"[red]Client {request.sid} disconnected[/red]")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

#Main
#Only if running from pure flask
#Ideally you want to run through some other WSGI server
#==================================================================================
if __name__ == "__main__":

    if not server_args.no_cache:
        console.print("Caching system prompt...")
        LLMSession.create(server_args.llm_backend).cache_system_prompt()
    
    socketio.run(app, port=server_args.port)
import os
import sys
sys.path.append(os.path.abspath(".."))

from flask import Flask, request, make_response, Response
from query_and_summarize.query import query as send_query
from rich.console import Console

console = Console()

app = Flask(__name__)

@app.route("/query")
def query():
    query_text = request.args.get("q", None)

    if query_text is None:
        console.print("[red]No query provided[/red]")
        resp = make_response("Please provide a query", 400)
        return resp
    else:
        #Create SSE stream
        return Response(map(lambda x: x.to_sse(), send_query(query_text)), mimetype='text/event-stream')

#Only if running from pure flask
if __name__ == "__main__":
    app.run(port=4625)
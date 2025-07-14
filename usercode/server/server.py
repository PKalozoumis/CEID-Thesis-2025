import os
import sys
sys.path.append(os.path.abspath("../.."))

from flask import Flask, request, make_response, Response
from query_code import query_function
from rich.console import Console
from classes import Arguments

console = Console()
app = Flask(__name__)

#==================================================================================

@app.route("/query")
def query():
    query_text = request.args.get("q", None)

    if query_text is None:
        console.print("[red]No query provided[/red]")
        resp = make_response("Please provide a query", 400)
        return resp

    #Create SSE stream
    return Response(query_function(
        query_text,
        sse_format=True,
        args=Arguments.from_query_params(request.args)
    ),
    mimetype='text/event-stream')

    #After a connection is terminated, the generator waits until the first yield to throw the exception
    #Only then can I actually stop the function
    
#Main
#Only if running from pure flask
#Ideally you want to run through some other WSGI server
#==================================================================================
if __name__ == "__main__":
    app.run(port=4625)
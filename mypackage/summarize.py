
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer, TextIteratorStreamer, TextStreamer
import sys
import os
from .elastic import elastic_session, ElasticDocument
from threading import Thread
from rich.live import Live
from rich.console import Console
from rich.panel import Panel

if __name__ == "__main__":
    model_path = "google/bigbird-pegasus-large-arxiv"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BigBirdPegasusForConditionalGeneration.from_pretrained(model_path)

    session = elastic_session("arxiv-index")
    doc = ElasticDocument(session, 325, filter_path="_source.article").get()

    inputs = tokenizer(doc, return_tensors='pt', truncation=True, max_length=4096)
    streamer = TextIteratorStreamer(tokenizer = tokenizer, decode_kwargs={'skip_special_tokens': True})
    #prediction = tokenizer.batch_decode(prediction)

    #Create new thread that will generate the arguments
    t = Thread(target=model.generate, kwargs={**inputs, 'streamer': streamer, 'num_beams': 1})
    t.start()

    #Render to the console
    #===================================================================================================

    console = Console()

    partial_text = ""
    panel = Panel(partial_text, title="Summary", border_style="cyan")

    with Live(panel, console=Console()) as live:
        for new_text in streamer:

            if new_text == ".<n>":
                new_text = "\n"

            partial_text += new_text
            panel.renderable = partial_text
            live.update(panel)

from ..elastic import Session, ElasticDocument
from ..cluster_selection import SelectedCluster
from ..helper import panel_print

from rich.live import Live
from rich.console import Console
from rich.panel import Panel

from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer, TextIteratorStreamer, TextStreamer, PegasusForConditionalGeneration
from threading import Thread
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

#================================================================================================================

def evaluate_summary_relevance(model: CrossEncoder, summary: str, query: str):
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L12-v2')
    tokens = tokenizer(summary)
    #print(tokens)

#================================================================================================================

def summarize(args, cluster: SelectedCluster):
    if args.cache:
        summary = load_summary(cluster)

    if not args.cache or summary is None:

        if args.m == "sus":
            #Let's try summarization of the entire cluster
            tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")
            summ_model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed")
            inputs = tokenizer(cluster.text, return_tensors='pt', truncation=True, max_length=4096)

            prediction = summ_model.generate(**inputs)
            prediction = tokenizer.batch_decode(prediction)

            panel_print(prediction)
            summary = prediction

        #----------------------------------------------------------------------------------------

        elif args.m == "llm":
            from mypackage.llm import llm_summarize
            
            full_text = ""
            removed_json = False

            #Retrieve fragments of text from the llm and add them to the full text
            #------------------------------------------------------------------------------
            with Live(panel_print(return_panel=True), refresh_per_second=10) as live:
                for fragment in llm_summarize(query.text, cluster.text):

                    full_text += fragment

                    #Clean up the json
                    #-------------------------------------------------
                    if fragment.endswith("}"):
                        full_text = full_text[:-2]

                    if not removed_json and full_text.startswith("{\"summary\": \""):
                        full_text = full_text[13:]
                        removed_json = True

                    #Once json is cleaned up, print
                    #-------------------------------------------------
                    else:
                        live.update(panel_print(full_text, return_panel=True))

            summary = full_text

        store_summary(selected_clusters[args.c], summary, args)

    #Evaluating the generated summary
    evaluate_summary_relevance(cross_model, summary, query.text)
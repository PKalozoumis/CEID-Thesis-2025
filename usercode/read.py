import os
import sys
sys.path.append(os.path.abspath(".."))

import argparse
from itertools import chain
import re

from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import panel_print, DEVICE_EXCEPTION
from mypackage.storage.load import load_pickles
from mypackage.cluster_selection import RelevanceEvaluator, SummaryCandidate
from mypackage.query import Query
from sentence_transformers import CrossEncoder

from rich.console import Console
from rich.rule import Rule
from rich.padding import Padding

console = Console()

#===============================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read the sentences and chains of a specific document")
    parser.add_argument("-i", action="store", type=str, help="The index name", default="pubmed")
    parser.add_argument("-d", action="store", type=str, help="Comma-separated list of document IDs")
    parser.add_argument("-x", action="store", type=str, help="Experiment")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cache", action="store_true", help="Use the cache")
    group.add_argument("--store", action="store_true", help="Store to cache")
    parser.add_argument("--print", action="store_true", help="Print full document text")
    parser.add_argument("-c", action="store", dest="chains", default="", type=str)
    parser.add_argument("--print-chains", action="store_true", default=True, dest="print_chains", help="Print chain text alongside its score")
    parser.add_argument("--no-print-chains", action="store_false", default=True, dest="print_chains", help="Print chain text alongside its score")
    parser.add_argument("-s", action="store", dest="sentences", default="", type=str)
    args = parser.parse_args()

    #--------------------------------------------------------------------------------------

    os.makedirs("cache", exist_ok=True)
    sess = Session(args.i, cache_dir=("cache" if args.cache else None), use="cache" if args.cache else "client")

    if args.d is None:
        raise DEVICE_EXCEPTION("BUT, THERE WAS NOTHING TO READ")
    
    if args.chains:
        query = Query(0, "What are the primary behaviours and lifestyle factors that contribute to childhood obesity", source=["summary", "article"], text_path="article")
        evaluator = RelevanceEvaluator(query, CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2'))
    
    docs = [ElasticDocument(sess, id=id, text_path="article") for id in args.d.split(",")]
    processed = load_pickles(sess, os.path.join("experiments", sess.index_name, "pickles", args.x), docs)

    #--------------------------------------------------------------------------------------

    for doc in processed:
        doc.doc.get()

        if args.print:
            panel_print(doc.doc.text, title=f"{doc.doc.id}")

        if args.sentences:
            to_print = []

            #Split the input string into the chains that we want to examine
            for sentence in args.sentences.split(","):
                sentence = int(sentence)
                to_print.append(Rule(f"[green]Sentence {sentence}[/green]"))
                to_print.append(Padding(doc.sentences[sentence].text, pad=(0,0,1,0)))
                
            panel_print(to_print)

        
        if args.chains:
            to_print = []

            #Split the input string into the chains that we want to examine
            for chain_group in args.chains.split(","):

                res = re.match(r"^(<*)(.*?)(>*)$", chain_group)
                left = len(res.group(1))
                right = len(res.group(3))
                chain_group = res.group(2)

                #Each chain can be a sum of chains, using the + symbol
                #This allows us to combine chains into a single text, for testing
                temp_chains = [doc.clustering.chains[int(chain_idx)] for chain_idx in chain_group.split("+")]

                candidate = SummaryCandidate.__new__(SummaryCandidate)
                candidate.evaluator = evaluator
                sc = round(evaluator.predict(temp_chains, join=True), 3)
                candidate.history = [SummaryCandidate.State(chains=temp_chains, score=sc)]           
                candidate.selected_state = -1
                candidate.expandable = True
                candidate.context.improvement_score = 0
                candidate.add_left_context(left)
                candidate.selected_state = -1
                candidate.add_right_context(right)
                candidate.selected_state = -1

                if args.print_chains: #Print text and score
                    to_print.append(Rule(f"[green]Chain {candidate.context.id}[/green]" + f" (size: {len(candidate.text.split())}, score: {candidate.score:.3f}), improvement score: {candidate.context.improvement_score:.3f}"))
                    to_print.append(Padding(candidate.text, pad=(0,0,1,0)))
                else: #Only score 
                    to_print.append(f"Chain [green]{chain_group}[/green]: [cyan]{sc:.3f}[/cyan]")

            panel_print(to_print)
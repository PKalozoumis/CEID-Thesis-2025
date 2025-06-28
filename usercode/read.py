import os
import sys
sys.path.append(os.path.abspath(".."))

import argparse
from itertools import chain

from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import panel_print, DEVICE_EXCEPTION
from mypackage.storage import load_pickles
from mypackage.cluster_selection import RelevanceEvaluator
from mypackage.query import Query
from sentence_transformers import CrossEncoder

from rich.console import Console
from rich.rule import Rule
from rich.padding import Padding

console = Console()

#===============================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read document contents from an index")
    parser.add_argument("-i", action="store", type=str, help="The index name", default="pubmed")
    parser.add_argument("-d", action="store", type=str, help="Comma-separated list of document IDs")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cache", action="store_true", help="Use the cache")
    group.add_argument("--store", action="store_true", help="Store to cache")
    parser.add_argument("--print", action="store_true", help="Disable printing")
    parser.add_argument("-chains", action="store", default="", type=str)
    parser.add_argument("--print-chains", action="store_true", default=True, dest="print_chains", help="Use the cross-encoder to evaluate the relevance of a chain to the query")
    parser.add_argument("--no-print-chains", action="store_false", default=True, dest="print_chains", help="Use the cross-encoder to evaluate the relevance of a chain to the query")
    parser.add_argument("--eval-chains", action="store_true", help="Use the cross-encoder to evaluate the relevance of a chain to the query")
    args = parser.parse_args()

    #--------------------------------------------------------------------------------------

    os.makedirs("cache", exist_ok=True)
    sess = Session(args.i, cache_dir=("cache" if args.cache else None), use="cache" if args.cache else "client")

    if args.d is None:
        raise DEVICE_EXCEPTION("BUT, THERE WAS NOTHING TO READ")
    
    if args.eval_chains:
        query = Query(0, "What are the primary behaviours and lifestyle factors that contribute to childhood obesity", source=["summary", "article"], text_path="article")
        evaluator = RelevanceEvaluator(query, CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2'))
    
    docs = [ElasticDocument(sess, id=id, text_path="article") for id in args.d.split(",")]
    processed = load_pickles(sess, os.path.join("experiments", sess.index_name, "pickles", "default"), docs)

    #--------------------------------------------------------------------------------------

    for doc in processed:
        doc.doc.get()

        if args.print:
            panel_print(doc.doc.text, title=f"{doc.doc.id}")
        
        if args.chains:
            to_print = []

            #Split the input string into the chains that we want to examine
            for chain_group in args.chains.split(","):

                #Each chain can be a sum of chains, using the + symbol
                #This allows us to combine chains into a single text, for testing
                temp_chains = [doc.clustering.chains[int(chain_idx)] for chain_idx in chain_group.split("+")]
                if args.eval_chains:
                    sc = round(evaluator.predict(temp_chains, join=True), 3)

                if args.print_chains: #Print text and score
                    if len(temp_chains) == 1: #Single chain
                        to_print.append(Rule(f"[green]Chain {chain_group}[/green]" + f" (size: {len(temp_chains[0].text.split())}" + (f", score: {sc:.3f})" if args.eval_chains else ")")))
                        to_print.append(Padding(temp_chains[0].text, pad=(0,0,1,0)))

                    else: #Concatenation of chains
                        temp_len = sum(len(c.text.split()) for c in temp_chains)
                        to_print.append(Rule(f"[green]Chain {chain_group}[/green]" + f" (size: {temp_len}" + (f", score: {sc:.3f})" if args.eval_chains else ")")))
                        to_print.append(Padding(" ".join((f"[cyan][{c.index}][/cyan]: {c.text}" for c in temp_chains)), pad=(0,0,1,0)))
                elif args.eval_chains: #Only score 
                    to_print.append(f"Chain [green]{chain_group}[/green]: [cyan]{sc:.3f}[/cyan]")

            panel_print(to_print)
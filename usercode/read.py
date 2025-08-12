'''
Print entire document
List specific sentences or chains, alongside their scores
Dynamically test the effects of chaining and context expansion
'''

import os
import sys
sys.path.append(os.path.abspath(".."))

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read the sentences and chains of a specific document")
    parser.add_argument("-i", action="store", type=str, help="The index name", default="pubmed")
    parser.add_argument("-d", action="store", type=str, help="Comma-separated list of document IDs")
    parser.add_argument("-x", action="store", type=str, help="Experiment", default="default")
    parser.add_argument("-db", action="store", type=str, default='mongo', help="Database to store the preprocessing results in", choices=['mongo', 'pickle'])

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cache", action="store_true", help="Use the cache")
    group.add_argument("--store", action="store_true", help="Store to cache")

    parser.add_argument("--print", action="store_true", help="Print full document text")
    parser.add_argument("--list-chains", action="store_true", default=False, help="Show chain sizes")
    parser.add_argument("--compare-sentence-embeddings", action="store_true", default=False, help="View sentence for the same document, but different experiments. For debugging")
    parser.add_argument("--compare-chains", action="store_true", default=False, help="Show chains for the same document, but many different experiments")

    parser.add_argument("-s", action="store", dest="sentences", default="", type=str)
    parser.add_argument("--score-sentences", action="store_true", default=False, help="Enable sentence evaluation")
    parser.add_argument("--print-sentences", action="store_true", default=True, dest="print_sentences", help="Print sentence text alongside its score")
    parser.add_argument("--no-print-sentences", action="store_false", default=True, dest="print_sentences", help="Print sentence text alongside its score")

    parser.add_argument("-c", action="store", dest="chains", default="", type=str, help="Comma-separated list of chains to read")
    parser.add_argument("--score-chains", action="store_true", default=False, help="Enable chain evaluation")
    parser.add_argument("--print-chains", action="store_true", default=True, dest="print_chains", help="Print chain text alongside its score")
    parser.add_argument("--no-print-chains", action="store_false", default=True, dest="print_chains", help="Print chain text alongside its score")    
    
    args = parser.parse_args()

#===============================================================================================================

from itertools import chain
import re

from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import panel_print, DEVICE_EXCEPTION
from mypackage.storage import DatabaseSession, MongoSession, PickleSession
from mypackage.cluster_selection import RelevanceEvaluator, SummaryCandidate
from mypackage.query import Query
from sentence_transformers import CrossEncoder
from mypackage.sentence import SimilarityPair

from rich.console import Console
from rich.rule import Rule
from rich.padding import Padding

console = Console()

#===============================================================================================================

if __name__ == "__main__":

    #--------------------------------------------------------------------------------------

    os.makedirs("cache", exist_ok=True)
    sess = Session(args.i, cache_dir=("cache" if args.cache else None), use="cache" if args.cache else "client")

    if args.db == "pickle":
        db = PickleSession(os.path.join("experiments", sess.index_name, "pickles"), args.x)
    else:
        db = MongoSession(db_name=f"experiments_{sess.index_name}", collection=args.x)

    if args.d is None:
        raise DEVICE_EXCEPTION("BUT, THERE WAS NOTHING TO READ")
    
    if args.score_chains:
        query = Query(0, "What are the primary behaviours and lifestyle factors that contribute to childhood obesity", source=["summary", "article"], text_path="article")
        evaluator = RelevanceEvaluator(query, CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2'))
    
    docs = [ElasticDocument(sess, id=id, text_path="article") for id in args.d.split(",")]
    #print(f"{db.base_path}/{db.sub_path}")
    processed = db.load(sess, docs)

    #--------------------------------------------------------------------------------------

    for doc in processed:
        doc.doc.get()

        #Print entire document
        if args.print:
            panel_print(doc.doc.text, title=f"{doc.doc.id}")
            sys.exit()

        if args.list_chains:
            console.print(doc.doc.chains)
            sys.exit()

        #==================================================================================================================

        #Print specific sentences
        if args.sentences:
            to_print = []

            if args.print_sentences and not args.score_sentences:
                for sentence in args.sentences.split(","):
                    sentence = int(sentence)
                    to_print.append(Rule(f"[green]Sentence {sentence}[/green]"))
                    to_print.append(Padding(doc.sentences[sentence].text, pad=(0,0,1,0)))

            elif args.score_sentences:

                #Split the input string into the chains that we want to examine
                for sentence in args.sentences.split(","):
                    res = re.match(r"^(.*?)(>?)$", sentence)
                    sentence_id = int(res.group(1))
                    right = len(res.group(2))

                    sim = 1
                    text = ""
                    title = ""
                
                    if right == 1:
                        s1 = doc.sentences[sentence_id]
                        s2 = doc.sentences[sentence_id+1]
                        sim = s1.similarity(s2)
                        text += s1.text + s2.text
                        title = f"[green]Sentences {s1.index}-{s2.index}[/green]"
                    else:
                        text = doc.sentences[sentence_id].text
                        title = f"[green]Sentence {sentence_id}[/green]"
                    
                    if args.print_sentences: #Print text and score
                        to_print.append(Rule(title + f" (size: {len(text.split())}, score: {sim:.3f})"))
                        to_print.append(Padding(text, pad=(0,0,1,0)))
                    else: #Only score 
                        to_print.append(f"{title}: [cyan]{sim:.3f}[/cyan]")
                
            panel_print(to_print)
        
        #==================================================================================================================

        if args.chains:
            to_print = []
            if args.print_chains and not args.score_chains:
                for chain_idx in args.chains.split(","):
                    ch = doc.clustering.chains[int(chain_idx)]
                    to_print.append(Rule(f"[green]Chain {chain_idx} ({ch.first_index} - {ch.last_index})[/green]"))
                    to_print.append(Padding(ch.text, pad=(0,0,1,0)))

            elif args.score_chains:
                #Split the input string into the chains that we want to examine
                for chain_group in args.chains.split(","):

                    res = re.match(r"^(<*)(.*?)(>*)$", chain_group)

                    left = len(res.group(1))
                    right = len(res.group(3))
                    chain_group = res.group(2)

                    #Each chain can be a sum of chains, using the + symbol
                    #This allows us to combine chains into a single text, for testing
                    temp_chains = [doc.clustering.chains[int(chain_idx)] for chain_idx in chain_group.split("+")]

                    #Make a fake candidate
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

    db.close()
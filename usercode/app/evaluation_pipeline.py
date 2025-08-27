import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, ElasticDocument
from mypackage.query import Query
from mypackage.summarization.metrics import bert_score, rouge_score
from mypackage.storage import DatabaseSession, MongoSession, PickleSession
from mypackage.cluster_selection.metrics import document_cross_score, document_cross_score_at_k
from mypackage.helper.retrieval_metrics import precision, recall, fscore, mean_average_precision, mean_reciprocal_rank, average_precision

from rich.console import Console

import pandas as pd

console = Console()

def retrieval_evaluation(sess: Session, query: Query, returned_docs: list[ElasticDocument], db: DatabaseSession, eval_relevance_threshold: float = 5.5):
    df = pd.read_csv(f"../dataset/query_{query.id}_results.csv")

    console.print(df.loc[df['score'] > eval_relevance_threshold])

    single_query_results = [doc.id for doc in returned_docs]
    relevant = df.loc[df['score'] > eval_relevance_threshold, 'doc'].to_list()

    console.print(f"Precision: {precision(single_query_results, relevant, vector=True)}")
    console.print(f"Recall: {recall(single_query_results, relevant, vector=True)}")
    console.print(f"F-Score: {fscore(single_query_results, relevant, vector=True)}")
    console.print(f"Average precision: {average_precision(single_query_results, relevant)}")
    console.print(f"MRR: {mean_reciprocal_rank([single_query_results], [relevant])}")

def evaluation_pipeline(sess: Session, query: Query, returned_docs: list[ElasticDocument], db: DatabaseSession, eval_relevance_threshold: float = 5.5):
    retrieval_evaluation(sess, query, returned_docs, db, eval_relevance_threshold)
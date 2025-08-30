import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, ElasticDocument
from mypackage.query import Query
from mypackage.summarization.metrics import bert_score, rouge_score, compression_ratio
from mypackage.storage import DatabaseSession, MongoSession, PickleSession
from mypackage.cluster_selection.metrics import document_cross_score, document_cross_score_at_k
from mypackage.helper.retrieval_metrics import precision, recall, fscore, mean_average_precision, mean_reciprocal_rank, average_precision
from mypackage.helper import panel_print, format_latex_table
from mypackage.summarization import SummaryUnit
from mypackage.cluster_selection import SelectedCluster, RelevanceEvaluator

from rich.console import Console

import time
import pandas as pd
import argparse

console = Console()

#=================================================================================================================

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
    
#=================================================================================================================

def cluster_selection_evaluation(returned_docs: list[ElasticDocument], selected_clusters: list[SelectedCluster], evaluator: RelevanceEvaluator, server_args: argparse.Namespace):
    t = time.time()
    score1 = document_cross_score(returned_docs, selected_clusters, evaluator, verbose=server_args.verbose, vector=True, keep_all_docs=False)
    console.print(score1)
    console.print(f"Score (filtered docs): {round(sum(score1)/len(score1), 3):.3f}")
    console.print(f"Evaluation time: {round(time.time() - t, 3):.3f}s\n")

    t = time.time()
    score2 = document_cross_score(returned_docs, selected_clusters, evaluator, verbose=server_args.verbose, vector=True, keep_all_docs=True)
    console.print(score2)
    x1 = sum(x for x, _ in score2)
    x2 = sum(x for _, x in score2)
    console.print(f"Score (all retrieved docs): {round(x1/x2, 3):.3f}")
    console.print(f"Evaluation time: {round(time.time() - t, 3):.3f}s")

    document_cross_score_at_k(score2)

#=================================================================================================================

def summary_evaluation(summaries: list[SummaryUnit]):
    for summary in summaries:
        panel_print(summary.reference, title="Reference")
        panel_print(summary.summary, title="Summary")
        #bert_score(summaries[0])
        rouge_score(summary)
        #compression_ratio(summary)

#=================================================================================================================

def evaluation_pipeline(
        sess: Session,
        query: Query,
        returned_docs: list[ElasticDocument],
        db: DatabaseSession,
        evaluator: RelevanceEvaluator,
        selected_clusters: list[SelectedCluster],
        summaries: list[SummaryUnit],
        *,
        server_args: argparse.Namespace,
        eval_relevance_threshold: float = 5.5
    ):

    #FOR TESTING ONLY 🗣
    #------------------------------------------
    if summaries[0].summary is None:
        tests = [
            "Childhood obesity is a complex issue influenced by various lifestyle factors and behaviours. Several lifestyle factors are associated with each other as behavioural contributors to childhood obesity <1923_99-99>. These factors often overlap, making it challenging to identify single causes.\n\nChildhood obesity has been linked to several immediate health risk factors, including orthopedic, neurological, pulmonary, gastroenterological, and endocrine conditions <272_6-7>. Furthermore, obesity has been indirectly associated with negative psychosocial outcomes in children, such as low self-esteem and depression, which can negatively impact academic performance and social relationships over time <272_6-7>. Lifestyle behaviours known to contribute significantly to childhood obesity include diet, physical activity, and stress. The consumption of sugar-sweetened beverages is an essential contributor to childhood obesity, confirmed by longitudinal studies <1923_61-63>.\n\nPhysical inactivity plays a major role in the rising prevalence of obesity, although excess energy intake also contributes <3611_66-68>. In fact, lack of physical activity is a leading cause of death worldwide and is particularly prevalent among women from AA and Hispanic populations <3611_66-68>. Food advertising on children's TV can facilitate adverse dietary patterns, such as energy-dense snack consumption and fast-food intake. Watching TV can also redirect attention away from conscious eating and provide opportunities for unnoticed snacking, which are risk factors for childhood obesity <1923_78-80>. The American Academy of Pediatrics policy statement suggests limiting daily screen time to mitigate these risks.\n\nWhile socioeconomic status (SES) has been linked to higher rates of childhood obesity, other factors such as family size, residence, and parental education do not appear to contribute significantly to the issue. However, rapid epidemiological transitions in metropolitan cities and peri-urban areas have led to an increase in prevalence among government school children <2581_65-66>. Interestingly, family stress levels may also be a contributing factor to childhood obesity, with obese children potentially increasing their parents' stress levels as well.",
            "Childhood obesity is a complex issue that arises from multiple factors. Several lifestyle behaviors, including diet, physical activity, and stress, are associated with an increased risk of childhood obesity <1923_99-99>. These behaviors often co-exist and can have far-reaching consequences on children's health and well-being.\n\nIn particular, a lack of physical activity is a major contributor to the rising prevalence of obesity among various populations, particularly among women, and is the fourth leading cause of death worldwide <3611_66-68>. Additionally, excessive energy intake through consuming energy-dense foods and sugary drinks also plays a significant role in childhood obesity <1923_61-63>. The consumption of sugar-sweetened beverages has been confirmed as an important contributor to childhood obesity by longitudinal studies <1923_61-63>. Furthermore, watching TV can facilitate adverse dietary patterns, such as energy-dense snack consumption and fast-food intake, which are linked to higher total energy intake and higher percentage energy from fat <1923_78-80>. Childhood obesity is also associated with various immediate health risk factors, including orthopedic, neurological, pulmonary, gastroenterological, and endocrine conditions. Moreover, obesity has been linked to negative psychosocial outcomes in children, such as low self-esteem and depression, which can indirectly affect academic performance and social relationships <272_6-7>. The stress level of the family can also be an important contributor to childhood obesity, and vice versa; obesity may increase the stress level <1923_89-89>. However, factors like family size, residence, and parent's education do not contribute significantly to obesity, although socioeconomic status (SES) is a notable factor, with obesity more prevalent in higher SES groups <2581_65-66>. In conclusion, childhood obesity is a multifaceted issue that requires a comprehensive approach to address the various lifestyle behaviors and factors contributing to it. Understanding these complexities is crucial for developing effective interventions and prevention strategies <1923_99-99><3611_66-68><272_6-7>.",
            "Childhood obesity is a complex issue influenced by various lifestyle factors and behaviours. Several lifestyle factors are associated with each other as contributors to childhood obesity, including diet, physical activity, and stress <1923_99-99>.\n\nImmediate health risk factors linked to childhood obesity include orthopedic, neurological, pulmonary, gastroenterological, and endocrine conditions, which can negatively impact children's psychosocial outcomes such as low self-esteem and depression <272_6-7>. Lifestyle behaviours known to be relevant in contributing to childhood obesity are diet, physical activity, and stress management <1923_5-5>. The European study on children demonstrated that surprisingly low proportions of them meet the recommended target values for these health behaviors <1923_61-63>, with consumption of sugar-sweetened beverages being a significant contributor.\n\nInactivity serves as a major role in rising obesity prevalence, alongside excess energy intake <3611_66-68>. Lack of physical activity is particularly notable among women and Hispanic populations. A sedentary lifestyle contributes to cardiovascular disease, hypertension, type 2 diabetes mellitus, obesity, multiple sclerosis, irritable bowel syndrome, and hyperlipidemia.\n\nPossible explanations for the childhood obesity epidemic involve multifold direct and indirect mechanisms <1923_78-80>. The most frequently advertised product category on children's TV is food, which facilitates adverse dietary patterns such as energy-dense snack consumption, fast-food consumption, and higher total energy intake. Watching TV can also redirect attention from conscious eating.\n\nAlthough stress level was not explicitly measured in the study, various other indicators suggest that family stress levels may be an important contributor to childhood obesity <1923_89-89>. Obesity can, in turn, increase the stress level of families. Additionally, studies have shown a steady increase in prevalence among government school children in large metropolitan cities and peri-urban areas, despite SES factors not contributing significantly to obesity <2581_65-66>."
        ]

        for i, s in enumerate(summaries):
            s.summary = tests[i]
    #------------------------------------------

    #retrieval_evaluation(sess, query, returned_docs, db, eval_relevance_threshold)
    #cluster_selection_evaluation(returned_docs, selected_clusters, evaluator, server_args)
    summary_evaluation(summaries)
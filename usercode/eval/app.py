'''
Evaluating the real-time component of the application
'''

import os
import sys
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath("../app"))

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("op", action="store", type=str, help="Operation to perform", choices=[
        "retrieval",
        "summ",
        "test",
        "times",
        "cluster_selection",
        "context_expansion"
    ])
    
    parser.add_argument("-i", "--index", action="store", type=str, default="pubmed", help="Comma-separated list of index names")
    parser.add_argument("-db", action="store", type=str, default='mongo', help="Database to load the preprocessing results from", choices=['mongo', 'pickle'])
    parser.add_argument("--cache", action="store_true", default=False, help="Retrieve docs from cache instead of elasticsearch")
    parser.add_argument("-f", "--files", required=True, action="store", type=str, help="Pickle files to load")
    parser.add_argument("--num-test-summaries", action="store", type=int, default=0, help="Number of test summaries to use. 0 for no test")
    parser.add_argument("-ndocs", "--num-documents", action="store", type=int, default=None, help="Number of retrieved docs to consider")
    parser.add_argument("--latex", action="store_true", default=False, help="Print latex tables")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    args = parser.parse_args()

from mypackage.elastic import Session, ElasticDocument
from mypackage.query import Query
from mypackage.summarization.metrics import bert_score, rouge_score, compression_ratio, rouge_presentation, bertscore_presentation, compression_presentation
from mypackage.storage import DatabaseSession, MongoSession, PickleSession, ExperimentManager, RealTimeResults
from mypackage.cluster_selection.metrics import document_cross_score, document_cross_score_at_k
from mypackage.helper.retrieval_metrics import precision, recall, fscore, mean_average_precision, mean_reciprocal_rank, average_precision, micro_avg_precision, micro_avg_recall, micro_avg_fscore, macro_avg_precision, macro_avg_fscore, macro_avg_recall
from mypackage.helper import panel_print, format_latex_table, DEVICE_EXCEPTION
from mypackage.summarization import SummaryUnit
from mypackage.cluster_selection import SelectedCluster, RelevanceEvaluator, print_candidates

from application_helper import create_time_tree

from itertools import chain
from rich.console import Console
from rich.rule import Rule
from rich.padding import Padding
import copy
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


from collections import defaultdict
import warnings

console = Console()

#=================================================================================================================

def load_real_time_results(sess: Session, db: DatabaseSession, exp_manager: ExperimentManager) -> tuple[list[list[RealTimeResults]], list[str]]:
    
    #Comma (,) separated eexperiments
    #Plus (+) separates queries for the same experiment
    #Each execution group corresponds to one or more executions (joint by +) for the same experiment, but different queries
    execution_groups: list[list[RealTimeResults]] = []
    experiment_names = [None]*len(args.files.split(","))

    #Gather the results of multiple executions
    for exp_num, group in enumerate(args.files.split(",")):
        exec_group: list[RealTimeResults] = []
        for file in group.split("+"):

            t = time.time()
            results = RealTimeResults.load(os.path.join("realtime_results", file), sess, db, exp_manager)
            if args.verbose: console.print(f"[green]Loaded results in[/green] [cyan]{round(time.time() - t, 3):.3f}s[/cyan]\n")

            experiment_names[exp_num] = results.args.experiment

            #Assign test summaries
            if args.num_test_summaries > 0 and results.summaries[0].summary is None:
                assign_test_summaries(results)

            #Cleanup summaries
            for sum in results.summaries:
                sum.clean_summary(no_citations=True, inplace=True)

            summaries = results.summaries

            if args.op == "summ" and summaries[0].summary is None:
                raise DEVICE_EXCEPTION("IT WAS ABRIDGED INTO NULLITY, AND CANNOT BE EVALUATED")
            
            exec_group.append(results)
        execution_groups.append(exec_group)

    return execution_groups, experiment_names

#=================================================================================================================

def test(multiple_exp_results: list[list[RealTimeResults]], names: list[str]):

    for exp_name, single_experiment_results in zip(names, multiple_exp_results):
        for q, single_query_results in enumerate(single_experiment_results):
            console.print(Padding(Rule(f"[red]Experiment {exp_name}, query {q}[/red]", style="red"), pad=(1,0,1,0)))
            #Print original clusters
            console.print(Rule("Original clusters"))
            for focused_cluster in single_query_results.original_selected_clusters:
                print_candidates(focused_cluster, title=f"Original candidates for cluster {focused_cluster.id}", current_state_only=True)

            #Print cluster selection results
            console.print(Rule("Final clusters"))
            for focused_cluster in single_query_results.selected_clusters:
                print_candidates(focused_cluster, title=f"Merged candidates for cluster {focused_cluster.id}", current_state_only=True)

            #Print text that was selected for summarization
            single_query_results.summaries[0].pretty_print(show_added_context=True, show_chain_indices=True, return_text=False)

            #Assign test summaries
            if args.num_test_summaries > 0 and single_query_results.summaries[0].summary is None:
                assign_test_summaries(single_query_results)

            #Raw input and output
            console.print(Rule("Summaries"))
            panel_print(single_query_results.summaries[0].reference, title="Reference")
            for i, s in enumerate(single_query_results.summaries):
                panel_print(s.summary, title="Summary" + (f" ({i+1}/{len(single_query_results.summaries)})" if len(single_query_results.summaries) > 1 else ""))

            #Print arguments and times
            panel_print(single_query_results.args.__dict__, title="Client args")
            tree, _ = create_time_tree(single_query_results.times[0])
            console.print(tree)

#=================================================================================================================

def retrieval_evaluation(multiple_exp_results: list[list[RealTimeResults]], eval_relevance_threshold: float = 5.3):
    
    #In retrieval evaluation, there are no multiple experiments (results are not affected by experiment)
    #So we just take the queries from the first experiment

    if len(multiple_exp_results) > 1:
        warnings.warn("When evaluating retrieval,multiple experiments are unnecessary. Only the first one will be considered.\n")

    multiple_query_results = []
    multiple_relevant = []

    for results in multiple_exp_results[0]:
        df = pd.read_csv(f"../retrieval_dataset/query_{results.query.id}_results.csv")

        single_query_results = [doc.id for doc in results.returned_docs][:args.num_documents]

        console.print(Rule())
        console.print(f"Retrieved docs: {single_query_results}")
        console.print("\nAll relevant docs:")
        console.print(df.loc[df['score'] > eval_relevance_threshold, ['doc', 'score']])
        relevant = df.loc[df['score'] > eval_relevance_threshold, 'doc'].to_list()

        multiple_query_results.append(single_query_results)
        multiple_relevant.append(relevant)

        print()

    #Show results for each of the 5 queries
    #-----------------------------------------------------------------------------

    query_records = []

    for results, relevant in zip(multiple_query_results, multiple_relevant):
        query_records.append({
            'Precision': precision(results, relevant),
            'Recall': recall(results, relevant),
            'F-Score': fscore(results, relevant),
            'RR': mean_reciprocal_rank([results], [relevant])
        })

    console.print(Rule())
    df = pd.DataFrame(query_records)
    console.print(df)

    if args.latex:
        latex = df.to_latex(
            escape=True,
            column_format='lcccc',
            caption="Μετρικές ανάκτησης για καθένα από τα ερωτήματα", 
            label="tab:retrieval_per_query", 
            float_format="%.3f",
            position="H"
        )
        format_latex_table(latex, name="Retrieval")
    print()

    #Show aggregate metrics
    #-----------------------------------------------------------------------------

    record = {
        'Precision': micro_avg_precision(multiple_query_results, multiple_relevant),
        'Recall': micro_avg_recall(multiple_query_results, multiple_relevant),
        'F-Score': micro_avg_fscore(multiple_query_results, multiple_relevant),
        'MAP': mean_average_precision(multiple_query_results, multiple_relevant),
        'MRR': mean_reciprocal_rank(multiple_query_results, multiple_relevant)
    }
    df = pd.DataFrame(record, index=[0])
    console.print(df)
    if args.latex:
        latex = df.to_latex(
            escape=True,
            column_format='lccccc',
            caption="Συσσωρευτικές μετρικές ανάκτησης για όλα τα ερωτήματα", 
            label="tab:retrieval_metrics", 
            float_format="%.3f",
            position="H"
        )
        format_latex_table(latex, name="Retrieval")

    '''
    console.print(f"Avg Precision: {micro_avg_precision(multiple_query_results, multiple_relevant, vector=True)}")
    console.print(f"Avg Recall: {micro_avg_recall(multiple_query_results, multiple_relevant, vector=True)}")
    console.print(f"Avg F-Score: {micro_avg_fscore(multiple_query_results, multiple_relevant, vector=True)}")
    console.print(f"MAP: {mean_average_precision(multiple_query_results, multiple_relevant)}")
    console.print(f"MRR: {mean_reciprocal_rank(multiple_query_results, multiple_relevant)}")
    '''

    #Show aggregate metrics
    #-----------------------------------------------------------------------------
    fig, ax = plt.subplots()

    metrics = np.array([
        micro_avg_precision(multiple_query_results, multiple_relevant, vector=True),
        micro_avg_recall(multiple_query_results, multiple_relevant, vector=True),
        micro_avg_fscore(multiple_query_results, multiple_relevant, vector=True)
    ])

    labels = ["Ακρίβεια", "Ανάκληση", "F-score"]
    markers = ['o', 'o', 'o']

    for data, label, marker in zip(metrics, labels, markers):
        ax.plot(list(range(1, len(data)+1)), data, label=label, marker=marker)

    ax.set_xlabel('Θέση κατάταξης')
    ax.set_ylabel('Τιμή')
    ax.grid(color="b", alpha=0.25)
    ax.legend()

    store_path = os.path.join(os.path.expanduser("~"), "ceid", "thesis-text", "images")
    fig.tight_layout(rect=[0,0,0.93,1])
    plt.savefig(f"{store_path}/retrieval.png", dpi=300)
    plt.show()
    
#=================================================================================================================

def cluster_selection_evaluation(multiple_exp_results: list[list[RealTimeResults]], names: list[str]):

    for exp_name, single_exp_results in zip(names, multiple_exp_results):
        for results in single_exp_results:
            cl_times = average_times(results.times)

            '''
            #Group cluster times by document
            time_per_doc = defaultdict(float)
            for k,v in cl_times.items():
                if k.startswith("context_expansion"):
                    doc = int(k.split("_")[2])
                    time_per_doc[doc] += v
            '''

            #cand_filter = results.args.cand_filter
            cand_filter = -6

            #I should check what percentage of chains in the initial clusters are actually good
            all_chain_count = sum(len(sc.central_chains()) for sc in results.original_selected_clusters)
            good_chain_count = sum(1 for sc in results.original_selected_clusters for cand in sc.candidates if cand.score > 0)
            chain_prec = round(good_chain_count/all_chain_count, 3)
            console.print(f"Chain precision: {chain_prec}")

            #We can check how good the initial clusters were at recalling information
            #Of course, we first need to filter out the negative chains
            #The idea is that we want to apply similar steps, as if this were the input to the summarization system
            #..but without context expansion
            filtered_clusters = [c.filter_and_merge_candidates(max_bridge_size=0) for c in results.original_selected_clusters]
            score, doc_times = document_cross_score(results.returned_docs, filtered_clusters, results.evaluator, verbose=args.verbose, cand_filter=cand_filter)
            console.print(f"Original Score: {score}")
            dt = sum(doc_times)
            
            score, _ = document_cross_score(results.returned_docs, results.selected_clusters, results.evaluator, verbose=args.verbose, cand_filter=cand_filter)
            console.print(f"Improved Score: {score}")

            ct = cl_times['cluster_retrieval'] + sum(v for k,v in cl_times.items() if k.startswith("context_expansion")) + sum(v for k,v in cl_times.items() if k.startswith("cross_score"))
            speedup = round(dt/ct, 2)
            console.print(f"Cluster processing time: {ct}")
            console.print(f"Document processing time: {dt}")
            console.print(f"Speedup: {speedup}")

            #document_cross_score_at_k(score2)

#=================================================================================================================

def context_expansion_evaluation(multiple_exp_results: list[list[RealTimeResults]], names: list[str]):
    raise NotImplementedError()
    
#=================================================================================================================

def summary_evaluation(multiple_exp_results: list[list[RealTimeResults]], names: list[str]):
    rouge_presentation(multiple_exp_results, names, show_latex=args.latex)
    bertscore_presentation(multiple_exp_results, names, show_latex=args.latex)
    compression_presentation(multiple_exp_results, names, show_latex=args.latex)

#=================================================================================================================

#FOR TEMPORARY TESTING ONLY
#IT DOES NOT MATTER WHAT PARAMETERS WERE THEORETICALLY USED TO GENERATE THESE
#WE ASSUME DEFAULT PARAMS, BECAUSE IT'S ONLY FOR TESTING
#(i will def forget this)
def assign_test_summaries(results: RealTimeResults):
    tests = [
        "Childhood obesity is a complex issue influenced by various lifestyle factors and behaviours. Several lifestyle factors are associated with each other as behavioural contributors to childhood obesity <1923_99-99>. These factors often overlap, making it challenging to identify single causes.\n\nChildhood obesity has been linked to several immediate health risk factors, including orthopedic, neurological, pulmonary, gastroenterological, and endocrine conditions <272_6-7>. Furthermore, obesity has been indirectly associated with negative psychosocial outcomes in children, such as low self-esteem and depression, which can negatively impact academic performance and social relationships over time <272_6-7>. Lifestyle behaviours known to contribute significantly to childhood obesity include diet, physical activity, and stress. The consumption of sugar-sweetened beverages is an essential contributor to childhood obesity, confirmed by longitudinal studies <1923_61-63>.\n\nPhysical inactivity plays a major role in the rising prevalence of obesity, although excess energy intake also contributes <3611_66-68>. In fact, lack of physical activity is a leading cause of death worldwide and is particularly prevalent among women from AA and Hispanic populations <3611_66-68>. Food advertising on children's TV can facilitate adverse dietary patterns, such as energy-dense snack consumption and fast-food intake. Watching TV can also redirect attention away from conscious eating and provide opportunities for unnoticed snacking, which are risk factors for childhood obesity <1923_78-80>. The American Academy of Pediatrics policy statement suggests limiting daily screen time to mitigate these risks.\n\nWhile socioeconomic status (SES) has been linked to higher rates of childhood obesity, other factors such as family size, residence, and parental education do not appear to contribute significantly to the issue. However, rapid epidemiological transitions in metropolitan cities and peri-urban areas have led to an increase in prevalence among government school children <2581_65-66>. Interestingly, family stress levels may also be a contributing factor to childhood obesity, with obese children potentially increasing their parents' stress levels as well.",
        "Childhood obesity is a complex issue that arises from multiple factors. Several lifestyle behaviors, including diet, physical activity, and stress, are associated with an increased risk of childhood obesity <1923_99-99>. These behaviors often co-exist and can have far-reaching consequences on children's health and well-being.\n\nIn particular, a lack of physical activity is a major contributor to the rising prevalence of obesity among various populations, particularly among women, and is the fourth leading cause of death worldwide <3611_66-68>. Additionally, excessive energy intake through consuming energy-dense foods and sugary drinks also plays a significant role in childhood obesity <1923_61-63>. The consumption of sugar-sweetened beverages has been confirmed as an important contributor to childhood obesity by longitudinal studies <1923_61-63>. Furthermore, watching TV can facilitate adverse dietary patterns, such as energy-dense snack consumption and fast-food intake, which are linked to higher total energy intake and higher percentage energy from fat <1923_78-80>. Childhood obesity is also associated with various immediate health risk factors, including orthopedic, neurological, pulmonary, gastroenterological, and endocrine conditions. Moreover, obesity has been linked to negative psychosocial outcomes in children, such as low self-esteem and depression, which can indirectly affect academic performance and social relationships <272_6-7>. The stress level of the family can also be an important contributor to childhood obesity, and vice versa; obesity may increase the stress level <1923_89-89>. However, factors like family size, residence, and parent's education do not contribute significantly to obesity, although socioeconomic status (SES) is a notable factor, with obesity more prevalent in higher SES groups <2581_65-66>. In conclusion, childhood obesity is a multifaceted issue that requires a comprehensive approach to address the various lifestyle behaviors and factors contributing to it. Understanding these complexities is crucial for developing effective interventions and prevention strategies <1923_99-99><3611_66-68><272_6-7>.",
        "Childhood obesity is a complex issue influenced by various lifestyle factors and behaviours. Several lifestyle factors are associated with each other as contributors to childhood obesity, including diet, physical activity, and stress <1923_99-99>.\n\nImmediate health risk factors linked to childhood obesity include orthopedic, neurological, pulmonary, gastroenterological, and endocrine conditions, which can negatively impact children's psychosocial outcomes such as low self-esteem and depression <272_6-7>. Lifestyle behaviours known to be relevant in contributing to childhood obesity are diet, physical activity, and stress management <1923_5-5>. The European study on children demonstrated that surprisingly low proportions of them meet the recommended target values for these health behaviors <1923_61-63>, with consumption of sugar-sweetened beverages being a significant contributor.\n\nInactivity serves as a major role in rising obesity prevalence, alongside excess energy intake <3611_66-68>. Lack of physical activity is particularly notable among women and Hispanic populations. A sedentary lifestyle contributes to cardiovascular disease, hypertension, type 2 diabetes mellitus, obesity, multiple sclerosis, irritable bowel syndrome, and hyperlipidemia.\n\nPossible explanations for the childhood obesity epidemic involve multifold direct and indirect mechanisms <1923_78-80>. The most frequently advertised product category on children's TV is food, which facilitates adverse dietary patterns such as energy-dense snack consumption, fast-food consumption, and higher total energy intake. Watching TV can also redirect attention from conscious eating.\n\nAlthough stress level was not explicitly measured in the study, various other indicators suggest that family stress levels may be an important contributor to childhood obesity <1923_89-89>. Obesity can, in turn, increase the stress level of families. Additionally, studies have shown a steady increase in prevalence among government school children in large metropolitan cities and peri-urban areas, despite SES factors not contributing significantly to obesity <2581_65-66>."
    ]

    original = results.summaries[0]
    results.summaries = []

    for s in tests[:args.num_test_summaries]:
        cp = copy.copy(original)
        cp.summary = s
        results.summaries.append(cp)

#=================================================================================================================

def average_times(times: list[dict]):
    # Initialize a defaultdict to sum values
    sums = defaultdict(float)

    for d in times:
        for k, v in d.items():
            sums[k] += v

    # Compute averages
    return {k: sums[k]/len(times) for k in sums}

def times(multiple_exp_results: list[list[RealTimeResults]], names: list[str]):
    rename_map = {
        "elastic": "Elastic",
        "query_encode": "Query Encode",
        "cluster_retrieval": "Cluster Retr.",
        "cross_scores": "Cross Scores",
        "context_expansion": "Context Exp.",
        "summary_time": "Summ.",
        "summary_response_time": "Resp. Time",
        "total": "Total"
    }

    '''
    rename_map = {
        "elastic": "Elastic",
        "query_encode": "Κωδικοπ. Ερωτ.",
        "cluster_retrieval": "Ανάκτηση Συστάδων",
        "cross_scores": "Cross Scores",
        "context_expansion": "Επέκταση Συμφρ.",
        "summary_time": "Σύνοψη",
        "summary_response_time": "Απόκρ. Σύνοψης",
        "total": "Σύνολο"
    }
    '''

    dfs = []

    for exp_name, single_experiment_results in zip(names, multiple_exp_results):
        experiment_times = []
        for single_query_results in single_experiment_results:
            experiment_times += single_query_results.times

        #For the default experiment only, also show the times for each of the 5 queries in detail
        #It is your responsibility to put the 5 queries in order lol
        if exp_name == "default":
            default_times = []
            for t in experiment_times:
                _, times_dict = create_time_tree(t, rename_map=rename_map)
                default_times.append(times_dict)
            default_times = pd.DataFrame(default_times)
            avg_row = pd.DataFrame([default_times.mean()], columns=default_times.columns, index=['average'])
            default_times = pd.concat([default_times, avg_row])

            if args.latex:
                latex = default_times.to_latex(
                    escape=True,
                    column_format='lXXXXXXXX',
                    caption="Times for each query", 
                    label="tab:default_times_per_query", 
                    float_format="%.3f",
                    position="H"
                )
                format_latex_table(latex, name="Times")

        console.print(Rule(f"Average times for Experiment {exp_name}"))
        tree, times_dict = create_time_tree(average_times(experiment_times), rename_map=rename_map)
        console.print(tree)
        dfs.append(pd.DataFrame(times_dict, index=[exp_name]))

    dfs = pd.concat(dfs, axis=0)
    console.print(dfs)

#=================================================================================================================

if __name__ == "__main__":
    sess = Session(args.index, base_path="../common", cache_dir="../cache", use="cache" if args.cache else "client")
    exp_manager = ExperimentManager("../common/experiments.json")
    db = DatabaseSession.init_db(args.db, exp_manager.db_name(sess.index_name))
    console.print()

    results, experiment_names = load_real_time_results(sess, db, exp_manager)

    match args.op:
        case "test": test(results, experiment_names)
        case "retrieval": retrieval_evaluation(results)
        case "summ": summary_evaluation(results, experiment_names)
        case "times": times(results, experiment_names)
        case "cluster_selection": cluster_selection_evaluation(results, experiment_names)
        case "context_expansion": context_expansion_evaluation(results, experiment_names)

        
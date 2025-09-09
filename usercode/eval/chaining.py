import os
import sys

sys.path.append(os.path.abspath("../.."))

import argparse

#=================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of preprocessing results")
    
    parser.add_argument("mode", nargs="?", action="store", type=str, help="Operation mode", choices=[
        "calculate",
        "present"
    ], default="calculate")
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs. Leave blank for a predefined set of test documents. -1 for all")
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("-i", "--index", action="store", type=str, default="pubmed", help="Index name")
    parser.add_argument("-db", action="store", type=str, default='mongo', help="Database to load the preprocessing results from", choices=['mongo', 'pickle'])
    parser.add_argument("-b", "--batch-size", action="store", type=int, default=200, help="Number of processed documents per batch")
    parser.add_argument("--cache", action="store_true", default=False, help="Retrieve docs from cache instead of elasticsearch")
    parser.add_argument("--columns", action="store_true", default=False, help="Experiments at the columns")
    parser.add_argument("--what", action="store", help="What to compare", choices=["sim@k", "centroid_sim@k", "length", "table"])
    parser.add_argument("--type", action="store", help="Comparison type (for sim_at_k and chain_length)")
    parser.add_argument("-m", "--metrics", action="store", type=str, required=False, help="Comma-separated metrics to present")
    parser.add_argument("--no-store", action="store_true", default=False, help="Do not store plots")
    args = parser.parse_args()

#=================================================================================================================

from collections import namedtuple
from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import NpEncoder, create_table, write_to_excel_tab, DEVICE_EXCEPTION, batched, format_latex_table, rule_print
from mypackage.sentence.metrics import chain_metrics, avg_within_chain_similarity, avg_chain_centroid_similarity, chaining_ratio
from mypackage.sentence import SentenceChain

import numpy as np
import pandas as pd

from mypackage.experiments import ExperimentManager
from mypackage.storage import PickleSession, MongoSession, DatabaseSession, ProcessedDocument

import pickle
from itertools import chain
from functools import reduce
import json

from rich.console import Console
from rich.rule import Rule
from rich.progress import track, Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from matplotlib import pyplot as plt

from multiprocessing import Pool
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

console = Console()
store_path = os.path.join(os.path.expanduser("~"), "ceid", "thesis-text", "images")

colors = ["#8fbbd3",  # light blue
          "#f8b455",  # light orange
          "#a4d37b",  # light green
          '#fb9a99',  # light red/pink
          '#cab2d6',  # light purple
          '#ffff99',  # light yellow
          '#fdbf6f',  # soft orange
          '#80b1d3',  # soft teal
          '#fccde5',  # soft pink
          '#d9d9d9']  # light gray


#================================================================================================

def gather_processed_batch(batch):
    workload = []
    for processed in db.load(sess, list(batch), skip_missing_docs=True, reject_list=exp_manager.rejected_docs(sess.index_name)):
        try:
            processed.doc.get()
            processed.doc.session = None
            workload.append(processed)
        except Exception:
            continue

    return workload

#================================================================================================

def sim_at_k_worker(doc: ProcessedDocument, sim_type: str) -> dict[int, float]:
    '''
    For a single document, return similarity scores of chains at each size k
    '''
    
    try:
        sizes = defaultdict(list[SentenceChain])
        for c in doc.chains:
            sizes[len(c)].append(c)

        max_chain_size = list(sizes.keys())[-1]
        #print(max_chain_size)
        score_at_k: dict[int, float] = {}
        for k in range(1, max_chain_size+1):

            if sim_type == "overall":
                temp = avg_within_chain_similarity(doc.chains, min_size=k, max_size=k, size_index=sizes)
            elif sim_type == "centroid":
                temp = avg_chain_centroid_similarity(doc.chains, vector=False, min_size=k, max_size=k, allow_self_similarity=True)
            if not np.isnan(temp):
                score_at_k[k] = temp

        return score_at_k
    except Exception as e:
        raise Exception(f"Document {doc.doc.id}") from e

#================================================================================================

def worker(doc: ProcessedDocument) -> dict:
    return {
        'doc': doc.doc.id,
        'sim_at_k': sim_at_k_worker(doc, "overall"),
        'centroid_sim_at_k': sim_at_k_worker(doc, "centroid"),
        'chain_length': [len(c) for c in doc.chains],
        'chain_centroid_sim': avg_chain_centroid_similarity(doc.chains, vector=True, min_size=2, allow_self_similarity=True),
        'chaining_ratio': chaining_ratio(doc.chains) 
    }

#================================================================================================

def calculate_metrics(docs: list[ElasticDocument], metrics = ["sim_at_k", "centroid_sim_at_k" "chain_length", "chain_centroid_sim", "chaining_ratio"]):

    similarity_at_k = defaultdict(list)
    centroid_similarity_at_k = defaultdict(list)
    records = defaultdict(list)

    with Pool(processes=5) as pool:
        batch_size = args.batch_size
        all_batches = list(batched(docs, batch_size))
        workload = gather_processed_batch(all_batches[0])
        task = progress.add_task(f"Experiment: [cyan]{exp}[/cyan]", start=True, total=len(docs))

        for i in range(len(all_batches)):
            #Distribute workload
            results =  pool.imap_unordered(worker, workload)

            #While work is being done, gather the next batch
            if i < len(all_batches)-1:
                workload = gather_processed_batch(all_batches[i+1])

            #Gather results
            for res in results:
                progress.update(task, advance=1)

                #Gather similarity at k
                if res['sim_at_k'] is not None:
                    for k,v in res['sim_at_k'].items():
                        similarity_at_k[k].append(v)

                #Gather centroid similarity at k
                if res['centroid_sim_at_k'] is not None:
                    for k,v in res['centroid_sim_at_k'].items():
                        centroid_similarity_at_k[k].append(v)

                records['chain_length'].append({'doc': res['doc'], 'chain_length': res['chain_length']})
                records['chain_centroid_sim'].append({'doc': res['doc'], 'chain_centroid_sim': res['chain_centroid_sim']})
                records['chaining_ratio'].append({'doc': res['doc'], 'chaining_ratio': res['chaining_ratio']})

    #Average scores per k
    #-------------------------------------------------------------------------
    if len(similarity_at_k) > 0:
        with open(f"metrics/{exp}/overall_sim_at_k.json", "w") as f:
            json.dump(dict(similarity_at_k), f, indent="\t")

    if len(centroid_similarity_at_k) > 0:
        with open(f"metrics/{exp}/centroid_sim_at_k.json", "w") as f:
            json.dump(dict(centroid_similarity_at_k), f, indent="\t")

    #Average lengths, centroid sims and chaining ratios
    #-------------------------------------------------------------------------
    with open(f"metrics/{exp}/chain_length.json", "w") as f:
        json.dump(records['chain_length'], f, indent="\t")
    with open(f"metrics/{exp}/chain_centroid_sim.json", "w") as f:
        json.dump(records['chain_centroid_sim'], f, indent="\t")
    with open(f"metrics/{exp}/chaining_ratio.json", "w") as f:
        json.dump(records['chaining_ratio'], f, indent="\t")

    progress.stop_task(task)

#=========================================================================================================

def similarity_at_k_presentation(comparison_type: str, similarity_type: str):
    dframes=[]

    match comparison_type:
        case "thres": experiments = "thres_55,default,thres_65,thres_70,thres_75"
        case "models": experiments = "default,mpnet"
        case "pooling": experiments = "default,most_similar_pooling"
        case "pooling2": experiments = "default,most_similar_pooling,mpnet_most_similar"
        case _: raise Exception

    for exp in db.available_experiments(experiments):
        with open(f"metrics/{exp}/{similarity_type}_sim_at_k.json") as f:
            similarity_at_k = json.load(f)

        for k in similarity_at_k:
            similarity_at_k[k] = np.nanmedian(similarity_at_k[k])

        keys = [int(k) for k in similarity_at_k.keys()]
        values = list(similarity_at_k.values())

        similarity_at_k = pd.DataFrame(values, index=keys).sort_index()
        similarity_at_k.columns = [exp]

        dframes.append(similarity_at_k)

    #I cannot draw merged df due to the line breaking at NaN values
    #I have to draw each df separately instead
    cmap = plt.cm.get_cmap("tab10").colors
    plt.figure(figsize=(12,7))
    plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.1)
    for i, df in enumerate(dframes):
        plt.plot(df.index, df.to_numpy(), marker='', linestyle='-', label=df.columns)
        db._sub_path = df.columns[0]
        params = db.get_experiment_params()
        plt.axhline(y=params['threshold'], color=cmap[i], linestyle='--')
    plt.legend()
    plt.grid(color="b", alpha=0.25)
    plt.xlabel("Μήκος αλυσίδας")
    plt.ylabel("Μέση ομοιότητα μεταξύ των προτάσεων" if similarity_type=="overall" else "Μέση ομοιότητα με αντιπρόσωπο")

    if not args.no_store:
        plt.savefig(f"{store_path}/{similarity_type}_sim_at_k_{comparison_type}.png", dpi=300)
    plt.show(block=True)

#=========================================================================================================

def chain_length_plot_presentation(comparison_type: str):

    df = None

    match comparison_type:
        case "thres": experiments = db.available_experiments("thres_55,default,thres_65,thres_70,thres_75")
        case "models": experiments = db.available_experiments("default,mpnet")
        case "pooling": experiments = db.available_experiments("default,most_similar_pooling")
        case _: raise Exception(f"Unknown comparison type {comparison_type}")

    if os.path.exists(f"metrics/{comparison_type}.parquet"):
        df = pd.read_parquet(f"metrics/{comparison_type}.parquet")

    if df is None:
        sizes_per_experiment = []
        for exp in experiments:
            with open(f"metrics/{exp}/chain_length.json", "r") as f:
                data = json.load(f)
                sizes_per_experiment.append(list(chain.from_iterable(d['chain_length'] for d in data)))

        df = pd.DataFrame({
            'Size': list(chain.from_iterable(sizes_per_experiment)),
            'Experiment': [f'{experiments[i]}' for i, exp in enumerate(sizes_per_experiment) for _ in exp]
        })

        df.to_parquet(f"metrics/{comparison_type}.parquet")

    #df = df[experiments]

    bucket_size = 7
    
    # Bucket sizes >= 15 into "15+"
    df['Size'] = df['Size'].apply(lambda x: x if x < bucket_size else f'{bucket_size}+')

     # Count number of groups per size for each method
    counts = df.groupby(['Experiment', 'Size'], sort=False).size().unstack(fill_value=0)

    # Reindex to preserve experiment order
    counts = counts.reindex(experiments, axis=0)

    # Ensure size order (1..bucket_size-1, then "bucket_size+")
    ordered_cols = list(range(1, bucket_size)) + [f'{bucket_size}+']
    counts = counts.reindex(columns=ordered_cols, fill_value=0)

    # Convert to percentages
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    # Plot bar plot
    ax = percentages.T.plot(kind='bar', width=0.7, color=[colors[i] for i in range(len(experiments))])
    fig = ax.get_figure()
    #fig.subplots_adjust(left=0.08,right=0.92, top=0.9)
    ax.set_xlabel('Μήκος')
    ax.set_ylabel('Ποσοστό συνολικών αλυσίδων')
    plt.xticks(rotation=0)
    plt.grid(color="b", alpha=0.25)

    if not args.no_store:
        plt.savefig(f"{store_path}/chain_length_{comparison_type}.png", dpi=300)
    plt.show()

#=========================================================================================================

def metric_to_df(exp: str, metric:str):
    

    if metric == "avg_chain_centroid_sim":
        micro_avg = False

        with open(f"metrics/{exp}/chain_centroid_sim.json", "r") as f: data = json.load(f)

        if micro_avg:
            temp = np.mean(list(chain.from_iterable(d['chain_centroid_sim'] for d in data)))
        else:
            temp = [np.mean(d['chain_centroid_sim']) for d in data if len(d['chain_centroid_sim']) > 0]
            temp = np.mean(temp)
    elif metric == "avg_chain_length":
        with open(f"metrics/{exp}/chain_length.json", "r") as f: data = json.load(f)
        temp = np.mean(list(chain.from_iterable(d['chain_length'] for d in data)))

    elif metric == "max_chain_length":
        with open(f"metrics/{exp}/chain_length.json", "r") as f: data = json.load(f)
        temp = np.max(list(chain.from_iterable(d['chain_length'] for d in data)))
        
    elif metric == "avg_chaining_ratio":
        with open(f"metrics/{exp}/chaining_ratio.json", "r") as f: data = json.load(f)
        temp = np.mean([d['chaining_ratio'] for d in data])
    
    else:
        raise Exception(f"Unknown metric {metric}")

    df = pd.DataFrame({'exp': exp, metric: temp}, index=["exp"]).set_index("exp")
    df.index.names = [None]
    return df


def table_presentation(experiments: list[str], metrics: list[str]):

    exp_dfs = []
    for exp in experiments:
        metric_dfs = []
        for metric in metrics:
            metric_dfs.append(metric_to_df(exp, metric))

        metric_dfs = pd.concat(metric_dfs, axis=1)[metrics]
        exp_dfs.append(metric_dfs)

    exp_dfs = pd.concat(exp_dfs, axis=0 if not args.columns else 1)
    console.print(exp_dfs)

    latex = exp_dfs.to_latex(
        escape=True,
        column_format='l' + 'l'*(len(metrics) if not args.columns else len(experiments)),
        caption="Μετρικές συσταδοποίησης", 
        label="tab:cluster", 
        float_format="%.3f",
        position="H"
    )
    format_latex_table(latex, name="Μετρικές συσταδοποίησης")


#=========================================================================================================

if __name__ == "__main__":

    if args.d == -1:
        raise Exception("no")

    exp_manager = ExperimentManager("../common/experiments.json")
    sess = Session(args.index, base_path="../common", cache_dir="../cache", use="cache" if args.cache else "client")
    docs = exp_manager.get_docs(args.d, sess, scroll_batch_size=2000)
    db = PickleSession() if args.db == "pickle" else MongoSession()
    db.base_path = os.path.join(sess.index_name, "pickles") if db.db_type == "pickle" else f"experiments_{sess.index_name}"

    if args.mode == "calculate":
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        ) as progress:

            experiments = db.available_experiments(args.x)
            for exp in experiments:
                os.makedirs(f"metrics/{exp}", exist_ok=True)
                db.sub_path = exp
                calculate_metrics(docs)
        
    elif args.mode == "present":

        match args.what:
            case "sim@k": similarity_at_k_presentation(args.type, "overall")
            case "centroid_sim@k": similarity_at_k_presentation(args.type, "centroid")
            case "length": chain_length_plot_presentation(args.type)
            case "table": table_presentation(args.x.split(","), args.metrics.split(","))

    db.close()
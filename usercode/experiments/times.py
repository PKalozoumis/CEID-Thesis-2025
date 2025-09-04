import sys
import os

sys.path.append(os.path.abspath("../.."))

import pandas as pd
import argparse
from rich.console import Console
import os
import re
from mypackage.helper import panel_print, rule_print, format_latex_table
from mypackage.elastic import ScrollingCorpus, Session
from itertools import groupby
from collections import defaultdict
import json
import re
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import math

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", action="store", type=str, help="CSV file name with processing times")
args = parser.parse_args()

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

#=============================================================================================================

def stats(df):
    agg = df.agg(['median', 'max', 'min', 'std']).round(3).astype(str)
    console.print(agg)
    rule_print(format_latex_table(agg.to_latex(
        escape=True,
        column_format='lrrrrr',
        caption="Χρόνοι",
        label="tab:processing_times",
        position="h"
    ),),
    title="Stats")

#=============================================================================================================

def time_bar_chart(dfs: list[pd.DataFrame]):
    
    labels = [str(i) for i in range(len(dfs))]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # keep same for sent_t, chain_t, cluster_t, umap_t
    
    # Aggregate and convert to ms
    df_ms_list = [(df.agg(['mean'])).round(3) for df in dfs]

    fig, ax = plt.subplots(figsize=(7,5))
    plt.subplots_adjust(left=0.08,right=0.92, top=0.9)
    #ax.grid(color="b", alpha=0.25)

    width = 0.5  # thinner bars
    x = np.arange(len(dfs))  # one x-position per df
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    for i, df_ms in enumerate(df_ms_list):
        bottom = np.zeros(len(df_ms))
        for j, col in enumerate(['sent_t', 'chain_t', 'cluster_t', 'umap_t']):
            ax.bar(x[i], df_ms[col].values[0], bottom=bottom, width=width, color=colors[j], label=col if i==0 else "")
            bottom += df_ms[col].values

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Median time [ms]')
    ax.set_title('Median Processing Time Comparison')
    ax.legend()
    plt.show()

#========================================================================================

def model_time_comparisons(dfs: list[pd.DataFrame]):
    
    labels = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    cols = ['sent_t', 'chain_t', 'cluster_t', 'umap_t']
    
    # Aggregate
    df_ms_list = [(df.agg(['mean'])).round(3) for df in dfs]

    fig, axes = plt.subplots(2, 2, figsize=(8,6))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    #colors = plt.cm.get_cmap("Set2").colors  # original
    #colors = [[(r+1)/2, (g+1)/2, (b+1)/2] for r, g, b in colors]  # lighten

    for ax, col, color in zip(axes.flatten(), cols, colors):
        values = [df[col].values[0] for df in df_ms_list]
        x = range(len(labels))
        ax.bar(x, values, color=color, width=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(col)
        ax.set_ylabel('Χρόνος (s)')
        ax.margins(x=0.2)
        
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(f"{store_path}/time_vs_model.png", dpi=300)
    plt.show()

#======================================================================

def threshold_chart(dfs: list[pd.DataFrame], labels):
    cols = dfs[0].columns.tolist()   # dynamic columns
    colors = plt.cm.tab10.colors     # auto color cycle
    
    # Aggregate
    df_ms_list = [(df.agg(['median'])).round(3) for df in dfs]

    n = len(cols)
    if n == 3:
        fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    elif n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
        axes = axes.flatten()
    else:
        raise ValueError("This function only supports DataFrames with 3 or 4 columns.")

    for ax, col, color in zip(axes, cols, colors):
        values = [df[col].values[0] for df in df_ms_list]
        ax.plot(labels, values, marker='o', color=color, label=col)
        ax.set_ylabel('Χρόνος (s)', fontsize=9)
        ax.set_title(col, fontsize=10)
        #ax.legend(fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(color="b", alpha=0.25)

    # X-label placement
    if n == 3:
        axes[-1].set_xlabel('Κατώφλι', fontsize=9)
    elif n == 4:
        axes[-2].set_xlabel('Κατώφλι', fontsize=9)
        axes[-1].set_xlabel('Κατώφλι', fontsize=9)

    #fig.suptitle('Χρόνος εκτέλεσης για διαφορετικά κατώφλια', fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(f"{store_path}/time_vs_threshold.png", dpi=300)
    plt.show()

#=============================================================================================================

def time_vs_doc_size(df: pd.DataFrame):
    #Get size of each doc in the collection
    sizes = {}
    if not os.path.exists("doc_sizes.json"):
        from mypackage.sentence.helper import split_to_sentences

        for doc in ScrollingCorpus(Session("pubmed", base_path="../common"), doc_field="article"):
            #sizes[doc.id] = len(doc.get().split())
            sizes[doc.id] = len(split_to_sentences(doc.get(), sep="\n"))
        with open("doc_sizes.json", "w") as f:
            json.dump(sizes, f)
    else:
        with open("doc_sizes.json", "r") as f:
            sizes = json.load(f)
            sizes = {int(k): v for k, v in sizes.items()}

    #Only keep sizes of docs that actually appear in the times df
    new_sizes = {}
    for _,doc in df.index.tolist():
        new_sizes[doc] = sizes[doc]
    sizes = new_sizes

    #Plot
    #-------------------------------------------------------------------
    doc_sizes = list(sizes.values())
    exec_times = df['sent_t'].tolist()

    plt.figure(figsize=(7, 5))
    plt.subplots_adjust(left=0.08,right=0.92, top=0.9)

    plt.scatter(doc_sizes, exec_times, label="Σημεία")

    # Regression line
    m, b = np.polyfit(doc_sizes, exec_times, 1)
    plt.plot(doc_sizes, np.array(doc_sizes)*m + b, color="red", label="Γραμμή τάσης")

    plt.xlabel("Πλήθος προτάσεων")
    plt.ylabel("Χρόνος εκτέλεσης (s)")
    plt.title(f"Χρόνος Εκτέλεσης vs Μέγεθος Κειμένου\n(Pearson r = {float(pearsonr(doc_sizes, exec_times).statistic):.3f}, p < 0.001)")
    plt.grid(color="b", alpha=0.25)
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/time_size_corr.png", dpi=300)
    plt.show()

#=============================================================================================================

def total_vs_sent_time(df: pd.DataFrame):
    sent_t = df['sent_t'].tolist()
    total_t = df['total'].tolist()

    plt.figure(figsize=(7, 5))
    plt.subplots_adjust(left=0.08,right=0.92, top=0.9)

    plt.scatter(sent_t, total_t, label="Σημεία")

    # Regression line
    m, b = np.polyfit(sent_t, total_t, 1)
    plt.plot(sent_t, np.array(sent_t)*m + b, color="red", label="Γραμμή τάσης")

    plt.xlabel("Χρόνος υπολογισμού ενσωματώσεων (s)")
    plt.ylabel("Συνολικός ρόνος εκτέλεσης (s)")
    plt.title(f"Συσχέτιση χρόνων εκτέλεσης (Pearson r = {float(pearsonr(sent_t, total_t).statistic):.3f}, p < 0.001)")
    plt.grid(color="b", alpha=0.25)
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/total_vs_sent_time.png", dpi=300)
    plt.show()


#=============================================================================================================    

def first_n(df: pd.DataFrame, n):
    
    df = df.iloc[:n].droplevel('exp')
    df.index.names = [None]
    latex = df.to_latex(
        escape=True,
        column_format='lXXXXX',
        caption="Τα πρώτα 10 κείμενα που επεξεργάζονται", 
        label="tab:first_n", 
        float_format="%.3f",
        position="h"
    )

    rule_print(format_latex_table(latex), title=f"First {n}")

#=============================================================================================================

def min_and_max_doc(df):
    max_idx = df.idxmax().apply(lambda x: x[1]).astype('str')
    max_val = df.max().astype('str')
    max_df = pd.DataFrame({
        'value': max_val,
        'doc': max_idx
    })

    min_idx = df.idxmin().apply(lambda x: x[1]).astype('str')
    min_val = df.min().astype('str')
    min_df = pd.DataFrame({
        'value': min_val,
        'doc': min_idx
    })

    arrays = [
        ['max']*len(max_df.columns) + ['min']*len(min_df.columns),
        list(max_df.columns) + list(min_df.columns)
    ]
            
    multi_index = pd.MultiIndex.from_arrays(arrays)

    combined_values = pd.concat([max_df, min_df], axis=1)
    combined_df = pd.DataFrame(combined_values.values, columns=multi_index, index=df.columns)


    console.print(combined_df)
    rule_print(format_latex_table(combined_df.to_latex(
        escape=True,
        multicolumn=True,
        column_format='lrrrrr',
        multicolumn_format="c",
        caption="Μέγιστες και ελάχιστες τιμές",
        label="tab:min_and_max",
        position="h"
    ),),
    title="Min and max documents")
    

def total_and_throughput(dfs: list[pd.DataFrame], df_names):
    result = pd.DataFrame(columns=df_names, index=['Συνολικός Χρόνος (s)', 'Ρυθμός Επεξ. (doc/s)'])
    for name, df in zip(df_names, dfs):
        total_time = df['total'].sum()
        num_docs = len(df)
        throughput = num_docs / total_time if total_time != 0 else 0
        result[name] = [total_time, throughput]
    result = result.round(3)
    
    console.print(result)

    latex = result.to_latex(
        escape=True,
        column_format='l' + 'l'*len(dfs),
        caption="Συνολικός χρόνος και ρυθμός επεξεργασίας", 
        label="tab:throughput", 
        float_format="%.3f",
        position="h"
    )

    rule_print(format_latex_table(latex), title=f"Συνολικός χρόνος και ρυθμός επεξεργασίας")

#=============================================================================================================

if __name__ == "__main__":
    preprocess_dir = os.path.join("..", "preprocess")
    if args.file is None:
        args.file = [f for f in sorted(os.listdir(preprocess_dir)) if re.match(r"preprocessing_results_(\d+)\.(\d+)\.csv", f)][-1]

    dfs = []
    for file in args.file.split(","):
        df = pd.read_csv(os.path.join(preprocess_dir, file), index_col=['exp', 'doc'])
        df['total'] = df.sum(axis=1)
        df = df.loc[df['umap_t']>0] #Some documents are so small they dont even get clustered

        #If the dataframe has multiple experiments, split them
        dfs.append(df)

    #console.print(df.loc[df['umap_t']==0])

    #--------------------------------------------------------------------------
    #console.print([df.agg(['median']).round(3) for df in dfs])
    console.print(dfs)
    
    total_and_throughput(dfs, df_names=["all-MiniLM-L6-v2", "all-mpnet-base-v2"][:len(dfs)])
    #threshold_chart([x[['chain_t', 'umap_t', 'cluster_t']].assign(total=lambda d: d.sum(axis=1)) for x in dfs],[0.55, 0.6, 0.65, 0.70, 0.75, 1])
    #model_time_comparisons([x[['sent_t', 'chain_t', 'umap_t', 'cluster_t']] for x in dfs])
    #time_vs_doc_size(dfs[0])
    #total_vs_sent_time(dfs[0])
    #stats(dfs[0])
    #min_and_max_doc(dfs[0])
    #first_n(dfs[0], 10)
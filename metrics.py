'''
Implementations of various IR metrics
'''
import math
from bisect import bisect_left
from matplotlib import pyplot as plt
import numpy
import pandas as pd
import os
import shutil
from collections import namedtuple

Score = namedtuple("Score", ["s1", "s2", "s3", "s4"])
Query = namedtuple("Query", ["id", "text", "num_results", "docs", "scores"])

#==============================================================================================

def gain_to_dcg(gain_vector: list[int]) -> list[float]:
    '''
    Input: Gain vector
    Output: Discounted cumulated gain vector
    '''

    dcg_vector = []

    for i, gain in enumerate(gain_vector):
        if i == 0:
            dcg_vector.append(gain)
        else:
            dcg_vector.append(dcg_vector[i-1] + gain/math.log(i + 1, 2))

    return dcg_vector

#==============================================================================================

def dcg(single_query_results: list[int], relevant: list[int], scores: list["Score"]) -> tuple[list, list]:
    '''
    **single_query_results**: List of documents the query returned\n
    **relevant**: List of relevant documents to the query\n
    **scores**: Scores of the respective relevant docs\n

    ## Returns
    - dcg_vector
    - idcg_vector
    '''
    #Calculate the gain
    gain_vector = []

    for doc in single_query_results:

        index = bisect_left(relevant, doc)

        if index == len(relevant) or relevant[index] != doc: #Document is not relevant
            gain_vector.append(0)
        else:
            score = scores[index]
            gain_vector.append(score.d1 + score.d2 +score.d3 +score.d4)

    ideal_gain_vector = sorted(gain_vector, reverse = True)

    return gain_to_dcg(gain_vector), gain_to_dcg(ideal_gain_vector)

#==============================================================================================

def ndcg(single_query_results: list[int], relevant: list[int], scores: list[Score]) -> list[float]:
    '''
    NDCG for a single query
    '''
    dcg_vector, idcg_vector = dcg(single_query_results, relevant, scores)

    return [a/b for a,b in zip(dcg_vector, idcg_vector)]

#==============================================================================================

def average_ndcg(multiple_query_results: list[list[int]], queries_dataset: list[Query]):

    num_queries = len(multiple_query_results)

    avg_dcg = [0 for _ in range(0, num_queries)]
    avg_idcg = [0 for _ in range(0, num_queries)]

    for q, single_query_results in zip(queries_dataset, multiple_query_results):
        dcg_vector, idcg_vector = dcg(single_query_results, q.docs, q.scores)

        avg_dcg = [a + b for a,b, in zip(avg_dcg, dcg_vector)]
        avg_idcg = [a + b for a,b, in zip(avg_idcg, idcg_vector)]

    return [a/b for a,b in zip(avg_dcg, avg_idcg)]

#==============================================================================================
    
def precision(single_query_results: list[int], relevant: list[int], /, vector=False):
    
    relevant_set = set(relevant)

    if vector:
        relevant_at_k = 0

        ret = []

        for k, res in enumerate(single_query_results):

            if res in relevant_set:
                relevant_at_k += 1

            ret.append(relevant_at_k/(k+1))

        return ret
    else:
        answer_set = set(single_query_results)
        return len(relevant_set & answer_set) / len(answer_set)
    
#==============================================================================================
    
def recall(single_query_results: list[int], relevant: list[int], /, vector=False):
    
    relevant_set = set(relevant)

    if vector:
        relevant_at_k = 0

        ret = []

        for res in single_query_results:

            if res in relevant_set:
                relevant_at_k += 1

            ret.append(relevant_at_k/len(relevant))

        return ret
    else:
        answer_set = set(single_query_results)
        return len(relevant_set & answer_set) / len(relevant_set)

#==============================================================================================
    
def fscore(single_query_results: list[int], relevant: list[int], /, vector=False) -> list:

    if vector:
        pak = precision(single_query_results, relevant, vector=True)
        rak = recall(single_query_results, relevant, vector=True)

        num_results = len(single_query_results)

        f = []

        for k in range(0, num_results):
            res = 0

            if pak[k] + rak[k] != 0:
                res = 2*pak[k]*rak[k]/(pak[k] + rak[k])

            f.append(res)

        return f
    else:
        p = precision(single_query_results, relevant)
        r = recall(single_query_results, relevant)

        res = 0

        if p + r != 0:
            res = 2*p*r/(p+r)

        return res

#==============================================================================================
    
def average_precision(single_query_results: list[int], relevant: list[int]):

    pak = precision(single_query_results, relevant, vector=True)

    recall_gain = 1/len(relevant)
    rak = [(recall_gain if res in relevant else 0) for res in single_query_results]

    return sum([a*b for a,b in zip(pak, rak)])

#==============================================================================================

def mean_average_precision(multiple_query_results: list[list[int]], queries_dataset: list[Query]):

    res = 0

    for i, single_query_results in enumerate(multiple_query_results):
        res += average_precision(single_query_results, queries_dataset[i])

    return res/len(multiple_query_results)

#==============================================================================================

def precision_at_k(single_query_results: list[int], relevant: list[int], k: int):
    relevant_set = set(relevant)

    top_k_results = single_query_results[:k]

    return len(top_k_results & relevant_set) / k

#==============================================================================================

def mean_reciprocal_rank(multiple_query_results: list[list[int]], queries_dataset: list[Query]):

    res = 0

    for i, single_query_results in enumerate(multiple_query_results):
        relevant = queries_dataset[i]["answers"]["docs"]

        for rank, result in enumerate(single_query_results):
            if result in relevant:
                res += 1/(rank + 1)
                break

    return res/len(multiple_query_results)

#==============================================================================================

def precision_recall_diagram(vsm_results, colbert_results, queries_dataset, query_ids: list, show_on_screen: bool):
    '''
    Creates a precision-recall diagram for the specified queries and saves them under plots/

    Parameters
    -   vsm_results: A list of lists, containing the search results of the Vector Space Model for every query
    -   colbert_results: A list of lists, containing the search results of ColBERT for every query
    -   queries_dataset: The dataset of queries, returned by load_datasets
    -   query_ids: A list of query IDs for which you want to make diagrams. Set None for all
    -   show_on_screen: Determines whether the plots should be shown on the screen on top of being saved
    '''

    if os.path.exists("results/plots/"):
        shutil.rmtree("results/plots/")

    os.makedirs("results/plots")

    data = {"Vector Space":[], "ColBERT": []}

    if query_ids is None:
        query_ids = range(1,len(queries_dataset)+1)

    for i, query_id in enumerate(query_ids):
        relevant = queries_dataset[query_id - 1]["answers"]["docs"]
        k = len(relevant)

        recall = [i/k for i in range(1, k+1)]

        #VSM
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        rel_count = 0
        total = 0

        precision = []

        for res in vsm_results[query_id - 1]:
            total += 1
            if res in relevant:
                rel_count += 1
                precision.append(rel_count/total)

        for _ in range(k - len(precision)):
            precision.append(0)

        area1 = numpy.trapz(precision, recall)

        fig, ax = plt.subplots()

        ax.plot(recall, precision, label="Vector Space")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall diagram for Query {query_id:02}")

        #Colbert
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        rel_count = 0
        total = 0

        precision = []

        for res in colbert_results[query_id - 1]:
            total += 1
            if res in relevant:
                rel_count += 1
                precision.append(rel_count/total)

        for _ in range(k - len(precision)):
            precision.append(0)

        area2 = numpy.trapz(precision, recall)

        ax.plot(recall, precision, label="ColBERT")

        ax.legend()

        #print(f"Query {query_id:02}\n================\nVSM: {round(area1, 3)}\nColBERT: {round(area2, 3)}\n")

        data["Vector Space"].append(round(area1, 3))
        data["ColBERT"].append(round(area2, 3))

        fig.savefig(f"results/plots/query_{query_id:02}.png")

        if i == len(query_ids) - 1:
            df = pd.DataFrame(data, index=query_ids)
            df.to_excel("results/precision_recall_area.xlsx", index_label="Query ID")

            if show_on_screen:
                plt.show(block = True)
        elif show_on_screen:
            plt.show(block = False)
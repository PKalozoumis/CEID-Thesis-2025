'''
Implementations of various IR metrics
'''
import math
from bisect import bisect_left

from ..query import Query

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

def relevance(doc_id: int, query: Query) -> str:
    index = bisect_left(query.docs, doc_id)

    if index == len(query.docs) or query.docs[index] != doc_id:
        return "0000"
    else:
        score = query.scores[index]
        return f"{score.s1}{score.s2}{score.s3}{score.s4}"

#===============================================================================================

def dcg(single_query_results: list[int], query: Query) -> tuple[list, list]:
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

        index = bisect_left(query.docs, doc)

        if index == len(query.docs) or query.docs[index] != doc: #Document is not relevant
            gain_vector.append(0)
        else:
            score = query.scores[index]
            gain_vector.append(score.s1 + score.s2 +score.s3 +score.s4)

    ideal_gain_vector = sorted(gain_vector, reverse = True)

    return gain_to_dcg(gain_vector), gain_to_dcg(ideal_gain_vector)

#==============================================================================================

def ndcg(single_query_results: list[int], query: Query) -> list[float]:
    '''
    NDCG for a single query
    '''
    dcg_vector, idcg_vector = dcg(single_query_results, query)

    return [round(a/b, 3) if b != 0 else 0 for a,b in zip(dcg_vector, idcg_vector)]

#==============================================================================================

def average_ndcg(multiple_query_results: list[list[int]], queries_dataset: list[Query]):

    num_queries = len(multiple_query_results)

    avg_dcg = [0 for _ in range(0, num_queries)]
    avg_idcg = [0 for _ in range(0, num_queries)]

    for q, single_query_results in zip(queries_dataset, multiple_query_results):
        dcg_vector, idcg_vector = dcg(single_query_results, q)

        avg_dcg = [a + b for a,b, in zip(avg_dcg, dcg_vector)]
        avg_idcg = [a + b for a,b, in zip(avg_idcg, idcg_vector)]

    return [round(a/b, 3) for a,b in zip(avg_dcg, avg_idcg)]

#==============================================================================================
    
def precision(single_query_results: list[int], relevant: list[int], /, vector=False):
    
    relevant_set = set(relevant)

    if vector:
        relevant_at_k = 0

        ret = []

        for k, res in enumerate(single_query_results):

            if res in relevant_set:
                relevant_at_k += 1

            ret.append(round(relevant_at_k/(k+1), 3))

        return ret
    else:
        answer_set = set(single_query_results)
        return round(len(relevant_set & answer_set) / len(answer_set), 3)
    
#==============================================================================================
    
def recall(single_query_results: list[int], relevant: list[int], /, vector=False):
    
    relevant_set = set(relevant)

    if vector:
        relevant_at_k = 0

        ret = []

        for res in single_query_results:

            if res in relevant_set:
                relevant_at_k += 1

            ret.append(round(relevant_at_k/len(relevant), 3))

        return ret
    else:
        answer_set = set(single_query_results)
        return round(len(relevant_set & answer_set) / len(relevant_set), 3)

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

            f.append(round(res, 3))

        return f
    else:
        p = precision(single_query_results, relevant)
        r = recall(single_query_results, relevant)

        res = 0

        if p + r != 0:
            res = round(2*p*r/(p+r), 3)

        return res

#==============================================================================================
    
def average_precision(single_query_results: list[int], relevant: list[int]):

    pak = precision(single_query_results, relevant, vector=True)

    recall_gain = 1/len(relevant)
    rak = [(recall_gain if res in relevant else 0) for res in single_query_results]

    return round(sum([a*b for a,b in zip(pak, rak)]), 3)

#==============================================================================================

def mean_average_precision(multiple_query_results: list[list[int]], queries_dataset: list[Query]):

    res = 0

    for i, single_query_results in enumerate(multiple_query_results):
        res += average_precision(single_query_results, queries_dataset[i].docs)

    return round(res/len(multiple_query_results), 3)

#==============================================================================================

def precision_at_k(single_query_results: list[int], relevant: list[int], k: int):
    relevant_set = set(relevant)

    top_k_results = single_query_results[:k]

    return round(len(top_k_results & relevant_set) / k, 3)

#==============================================================================================

def mean_reciprocal_rank(multiple_query_results: list[list[int]], multiple_relevant: list[list[int]]):

    res = 0

    for i, single_query_results in enumerate(multiple_query_results):
        relevant = multiple_relevant[i]

        for rank, result in enumerate(single_query_results):
            if result in relevant:
                res += 1/(rank + 1)
                break

    return round(res/len(multiple_query_results), 3)

#==============================================================================================
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage import RealTimeResults

import evaluate
from transformers import logging
from .classes import SummaryUnit
from rouge_score import rouge_scorer
from ..helper import panel_print, format_latex_table

import pandas as pd

import torch
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
from bert_score import score as _bert_score

from rich.console import Console


logging.set_verbosity_error()  # hides warnings and info messages
console = Console()

#=====================================================================================================

def bert_score(summary_unit: SummaryUnit):
    P, R, F1 = _bert_score([summary_unit.summary], [summary_unit.reference], lang="en", model_type="roberta-large")
    P = P.mean().item()
    R = R.mean().item()
    F1 = F1.mean().item()

    # Build a DataFrame with metrics as rows
    df = pd.DataFrame({
            "precision": P,
            "recall": R,
            "fmeasure": F1}, index=["BERT-Score"])

    return df

#=====================================================================================================

def rouge_score(summary_unit: SummaryUnit):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    reference = summary_unit.reference
    candidate = summary_unit.summary
    scores = scorer.score(reference, candidate)

    # Build a DataFrame with metrics as rows
    df = pd.DataFrame({
        metric: {
            "precision": scores[metric].precision,
            "recall": scores[metric].recall,
            "fmeasure": scores[metric].fmeasure
        }
        for metric in scores
    }).T  # transpose so metrics become rows

    return df

#=====================================================================================================

def compression_ratio(summary_unit: SummaryUnit):
    source_len = len(summary_unit.reference.split())
    summary_len = len(summary_unit.summary.split())

    compression_ratio = summary_len / source_len
    df = pd.DataFrame({"compression_ratio": compression_ratio}, index=[0])

    return df

#=====================================================================================================

def _gather_scores(metric, execution_groups: list[list[RealTimeResults]]) -> list[pd.DataFrame]:
    '''
    If overthinking were a function
    '''
    full_scores = []
    for ex_group in execution_groups:
        #An ex_group represents a series of executions for the same experiment, but different queries
        #We gather the scores across all queries and summaries of the same experiment in the group_scores list
        #Then, we average them
        group_scores = []
        for query_ex in ex_group:
            #query_ex here represents a set of summaries that were generated from the same query and experiment
            #For this specific query of the group, we gather those summary scores and add them to the overall group scores
            group_scores += [metric(s) for s in query_ex.summaries]

        #For ONE experiment, we have gathered all the results across all queries and their multiple summaries
        #Now we average
        group_scores = pd.concat(group_scores, axis=0).groupby(level=0).mean()
        full_scores.append(group_scores)

    return full_scores

#=====================================================================================================

def rouge_presentation(multiple_exp_results: list[list[RealTimeResults]], names: list[str], show_latex: bool = False):
    full_scores = _gather_scores(rouge_score, multiple_exp_results)

    if len(full_scores) == 1:
        console.print(full_scores[0])
        if show_latex:
            latex = full_scores[0].to_latex(
                escape=True,
                column_format='lccc',
                caption="Rouge scores", 
                label="tab:rouge", 
                float_format="%.3f",
                position="h"
            )
            format_latex_table(latex, name="Rouge")
    else:
        merged_compact = []
        for df in full_scores:
            row = {}
            for metric in df.index:
                p, r, f = df.loc[metric, ["precision", "recall", "fmeasure"]]
                decimals = 3
                row[metric] = f"{p:.{decimals}f} / {r:.{decimals}f} / {f:.{decimals}f}"
            merged_compact.append(row)

        full_scores = pd.DataFrame(merged_compact)
        full_scores.index = names
        #full_scores.index.name = "Experiment"
        full_scores.columns = ["ROUGE-1 (p/r/f)", "ROUGE-2 (p/r/f)", "ROUGE-L (p/r/f)"]
        console.print(full_scores)

        if show_latex:
            latex = full_scores.to_latex(
                escape=True,
                column_format='lccc',
                caption="Rouge scores", 
                label="tab:rouge", 
                float_format="%.3f",
                position="h"
            )
            format_latex_table(latex, name="Rouge")

#=====================================================================================================

def bertscore_presentation(multiple_exp_results: list[list[RealTimeResults]], names: list[str], show_latex: bool = False):
    full_scores = _gather_scores(bert_score, multiple_exp_results)

    if len(full_scores) == 1:
        console.print(full_scores[0])
        if show_latex:
            latex = full_scores[0].to_latex(
                escape=True,
                column_format='lccc',
                caption="Bert scores", 
                label="tab:bert_scores", 
                float_format="%.3f",
                position="h"
            )
            format_latex_table(latex, name="BERT-Score")
    else:
        full_scores = pd.concat(full_scores, axis=0).reset_index(drop=True)
        full_scores.index = names
        #full_scores.index.name = "Experiment"
        console.print(full_scores)

        if show_latex:
            latex = full_scores.to_latex(
                escape=True,
                column_format='lccc',
                caption="BERT scores", 
                label="tab:bert_scores", 
                float_format="%.3f",
                position="h"
            )
            format_latex_table(latex, name="BERT-Score")

#=====================================================================================================

def compression_presentation(multiple_exp_results: list[list[RealTimeResults]], names: list[str], show_latex: bool = False):
    full_scores = _gather_scores(compression_ratio, multiple_exp_results)

    if len(full_scores) == 1:
        console.print(full_scores[0])
        if show_latex:
            latex = full_scores[0].to_latex(escape=True,column_format='lccc',caption="Compression ratios", 
                label="tab:compression", 
                float_format="%.3f",
                position="h"
            )
            format_latex_table(latex, name="Compression")
    else:
        full_scores = pd.concat(full_scores, axis=0).reset_index(drop=True)
        full_scores.index = names
        #full_scores.index.name = "Experiment"
        console.print(full_scores)

        if show_latex:
            latex = full_scores.to_latex(escape=True,column_format='lccc',caption="Compression ratios", label="tab:compression", float_format="%.3f",position="h")
            format_latex_table(latex, name="Compression")
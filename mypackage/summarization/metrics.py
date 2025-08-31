import evaluate
from transformers import AutoTokenizer
from .classes import SummaryUnit
from rouge_score import rouge_scorer
from ..helper import panel_print, format_latex_table

import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
from bert_score import score as _bert_score

from rich.console import Console

console = Console()

#=====================================================================================================

def bert_score(summary_unit: SummaryUnit):
    P, R, F1 = _bert_score([summary_unit.summary], [summary_unit.reference], lang="en", model_type="roberta-large")
    print("Precision:", P.mean().item())
    print("Recall:", R.mean().item())
    print("F1:", F1.mean().item())

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
    print(f"Compression ratio: {compression_ratio:.2f}")

#=====================================================================================================

'''
def meteor_score(summary_unit: SummaryUnit):
    reference_tokens = word_tokenize(summary_unit.reference)
    candidate_tokens = word_tokenize(summary_unit.summary)

    # Compute METEOR
    score = single_meteor_score(reference_tokens, candidate_tokens)
    print(f"METEOR score: {score:.4f}")
'''
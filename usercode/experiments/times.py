import sys
import os

sys.path.append(os.path.abspath("../.."))

import pandas as pd
import argparse
from rich.console import Console
import os
import re
from mypackage.helper import panel_print, rule_print
from itertools import groupby

import re

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", action="store", type=str, help="CSV file name with processing times")
args = parser.parse_args()

console = Console()

#=============================================================================================================

import re

def format_latex_table(latex_code: str) -> str:
    # Insert caption formatting before \caption
    latex_code = re.sub(
        r"(\\caption\{)",
        r"\\captionsetup{font=normalsize, labelfont=bf}\n\1",
        latex_code,
        count=1
    )

    # Extract tabular spec
    match = re.search(r"\\begin\{tabular\}\{(.*?)\}", latex_code, re.DOTALL)
    if not match:
        return latex_code
    
    col_spec = match.group(1)

    # If there is an X → use tabularx
    if "X" in col_spec:
        new_cols = []
        count = 0
        last_char = None

        for c in col_spec:
            if c == "X":
                if last_char == "X":
                    count += 1
                else:
                    if last_char:
                        if last_char == "X":
                            new_cols.append(rf"*{{{count}}}{{>{{\\raggedleft\\arraybackslash}}X}}")
                        else:
                            new_cols.append(last_char * count)
                    count = 1
                last_char = "X"
            else:
                if last_char == "X":
                    new_cols.append(rf"*{{{count}}}{{>{{\\raggedleft\\arraybackslash}}X}}")
                    count = 0
                if last_char == c:
                    count += 1
                else:
                    if last_char and last_char != "X":
                        new_cols.append(last_char * count)
                    count = 1
                last_char = c

        # flush last group
        if last_char == "X":
            new_cols.append(rf"*{{{count}}}{{>{{\\raggedleft\\arraybackslash}}X}}")
        else:
            new_cols.append(last_char * count)

        new_spec = "".join(new_cols)

        latex_code = re.sub(
            r"\\begin\{tabular\}\{.*?\}",
            r"\\begin{tabularx}{\\textwidth}{" + new_spec + "}",
            latex_code,
            count=1
        )
        latex_code = re.sub(r"\\end\{tabular\}", r"\\end{tabularx}", latex_code, count=1)

    else:
        # No X → just center it
        latex_code = re.sub(
            r"(\\begin\{tabular\}\{.*?\})",
            r"\\centering\n\1",
            latex_code,
            count=1
        )

    return latex_code

    

#=============================================================================================================

def stats(df):
    agg = df.agg(['min', 'max', 'median', 'std']).round(3).astype(str)
    rule_print(format_latex_table(agg.to_latex(
        escape=True,
        column_format='lrrrrr',
        caption="Χρόνοι",
        label="tab:processing_times",
        position="h"
    ),),
    title="Stats")

#=============================================================================================================    

def first_n(df: pd.DataFrame, n):
    
    df = df.iloc[:n].droplevel('exp')
    df.index.names = [None]
    latex = df.to_latex(
        escape=True,
        column_format='lXXXXX',
        caption="Processing times per stage", 
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

    console.print(max_df)

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
    

#=============================================================================================================

if __name__ == "__main__":
    preprocess_dir = os.path.join("..", "preprocess")
    if args.file is None:
        args.file = [f for f in sorted(os.listdir(preprocess_dir)) if re.match(r"preprocessing_results_(\d+)\.(\d+)\.csv", f)][-1]

    df = pd.read_csv(os.path.join(preprocess_dir, args.file), index_col=['exp', 'doc'])
    df['total'] = df.sum(axis=1)

    df = df.loc[df['umap_t']>0]

    #--------------------------------------------------------------------------

    stats(df)
    min_and_max_doc(df)
    first_n(df, 10)
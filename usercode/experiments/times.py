import pandas as pd
import argparse
from rich.console import Console
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", action="store", type=str, help="CSV file name with processing times")
args = parser.parse_args()

console = Console()

if __name__ == "__main__":
    preprocess_dir = os.path.join("..", "preprocess")
    if args.file is None:
        args.file = [f for f in sorted(os.listdir(preprocess_dir)) if re.match(r"preprocessing_results_(\d+)\.(\d+)\.csv", f)][-1]

    df = pd.read_csv(os.path.join(preprocess_dir, args.file), index_col=['exp', 'doc'])
   
    df['total'] = df.sum(axis=1)
    console.print(df)

    console.print(f"Max: {df.max()}")

    console.print(df.sort_values(by="sent_t", axis=0, ascending=False).iloc[20:40])
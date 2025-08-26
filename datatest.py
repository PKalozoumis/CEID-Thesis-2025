from datasets import load_dataset
from rich.console import Console

console = Console()

dataset = load_dataset('Shitao/MLDR', "en")

for x in dataset['train']:
    console.print(f"<{x['query_id']}>: \"{x['query']}\" ({len(x['positive_passages'])} passages)")

#console.print(dataset['dev'][7])

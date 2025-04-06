from mypackage.collection_helper import generate_examples
import os
import json

file_path = os.path.join("collection", "pubmed.txt")
ex = generate_examples(file_path, doc_limit=20)

for i, doc in enumerate(ex):
    with open(f"cached_docs/pubmed_{i:04}.json", "w") as f:
        json.dump(doc, f)
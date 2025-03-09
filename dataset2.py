import json
import sys
import time

def read_next_file():
    with open("arxiv-metadata-oai-snapshot.json") as f:
        for line in f:
            metadata = json.loads(line)
            yield 


for data in read_next_file():
    print(data, end="\n\n")
    time.sleep(1)
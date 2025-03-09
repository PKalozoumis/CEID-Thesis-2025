
import json
import sys
import time

def read_next_file():
    with open("train.json") as f:
        for obj in json.load(f):
            yield obj 


for i, data in enumerate(read_next_file()):
    print(f"{i}: {data}", end="\n\n")
    time.sleep(0.5)
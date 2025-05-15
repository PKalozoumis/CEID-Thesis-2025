from collections import namedtuple
import pickle
import os
import json

ProcessedDocument = namedtuple("ProcessedDocument", ["doc", "chains", "labels", "clusters"])

#=============================================================================================================

def load_pickles(experiment: str, docs, index_name) -> ProcessedDocument | list[ProcessedDocument]:
    pkl = []

    if type(docs) is int:
        temp = [docs]
    else:
        temp = docs

    for fname in map(lambda x: os.path.join(index_name, "pickles", experiment, f"{x}.pkl"), temp):
        with open(fname, "rb") as f:
            pkl.append(pickle.load(f))

    if type(docs) is int:
        return pkl[0]
    else:
        return pkl
    
#=============================================================================================================

def load_experiment(experiments: str|list[str]|None) -> dict | list[dict]:
    with open("experiments.json", "r") as f:
        temp = json.load(f)

    default_experiment = temp['default'] | {'name': 'default'}

    if type(experiments) is str:
        if experiments in temp:
            return default_experiment | temp[experiments] | {'name': experiments}
        else:
            print(f"Could not find experiment '{experiments}'. Using default params")
            return default_experiment
    elif isinstance(experiments, list):
        return [default_experiment | v | {'name': k} for k,v in temp.items() if k in experiments]
    elif experiments is None:
        return [default_experiment | v | {'name': k} for k,v in temp.items()]

#=============================================================================================================

def experiment_generator():
    with open("experiments.json", "r") as f:
        experiments = json.load(f)

    default_experiment = experiments['default'] | {'name': 'default'}

    for xp_name in experiments:
        yield default_experiment | experiments[xp_name] | {'name': xp_name}

#=============================================================================================================

def experiment_wrapper(name):
    if name == "all":
        return experiment_generator()
    else:
        return load_experiment(name.split(','))
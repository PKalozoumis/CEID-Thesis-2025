from collections import namedtuple
import pickle
import os
import json

#=============================================================================================================

def load_experiments(experiment_names: str|list[str]|None = None, must_exist: bool = False) -> dict | list[dict]:

    if isinstance(experiment_names, list) and len(experiment_names) == 1:
        experiment_names = experiment_names[0]

    with open("experiments.json", "r") as f:
        temp = json.load(f)

    default_experiment = temp['default'] | {'name': 'default'}
    if type(experiment_names) is str:
        
        if experiment_names in temp:
            return default_experiment | temp[experiment_names] | {'name': experiment_names}
        else:
            if must_exist:
                raise Exception("THIS NEXT EXPERIMENT SEEMS... VACANT")
            else:
                print(f"Could not find experiment '{experiment_names}'. Using default params")
                return default_experiment | {'name': "default"}
    elif isinstance(experiment_names, list):
        return [default_experiment | v | {'name': k} for k,v in temp.items() if k in experiment_names]
    elif experiment_names is None:
        return [default_experiment | v | {'name': k} for k,v in temp.items()]

#=============================================================================================================

def all_experiments():
    with open("experiments.json", "r") as f:
        experiments = json.load(f)

    default_experiment = experiments['default'] | {'name': 'default'}

    for xp_name in experiments:
        yield default_experiment | experiments[xp_name] | {'name': xp_name}

#=============================================================================================================

def experiment_wrapper(experiment_names: str | list[str], must_exist: bool = False, strict_iterable: bool = True):
    if experiment_names == "all":
        return list(all_experiments())
    else:
        xp = load_experiments(experiment_names, must_exist)
        if not strict_iterable or isinstance(xp, list):
            return xp
        else:
            return [xp]
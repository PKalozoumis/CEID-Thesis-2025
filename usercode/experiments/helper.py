import os
import sys
sys.path.append("../..")

from mypackage.helper import DEVICE_EXCEPTION
import os
import json

CHOSEN_DOCS = {
    'pubmed': [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 1106]
}

#=============================================================================================================

def document_index(index_name: str, doc_id, fallback: int = None) -> int:
    try:
        return CHOSEN_DOCS.get(index_name, list(range(10))).index(doc_id)
    except ValueError:
        return fallback

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
                raise DEVICE_EXCEPTION("THIS NEXT EXPERIMENT SEEMS... VACANT")
            else:
                print(f"Could not find experiment '{experiment_names}'. Using default params")
                return default_experiment | {'name': "default"}
    elif isinstance(experiment_names, list):
        return [default_experiment | v | {'name': k} for k,v in temp.items() if k in experiment_names]
    elif experiment_names is None:
        return [default_experiment | v | {'name': k} for k,v in temp.items()]

#=============================================================================================================

def all_experiments(*,names_only=False):
    with open("experiments.json", "r") as f:
        experiments = json.load(f)

    if names_only:
        yield from experiments
    else:
        default_experiment = experiments['default'] | {'name': 'default'}

        for xp_name in experiments:
            yield default_experiment | experiments[xp_name] | {'name': xp_name}

#=============================================================================================================

def experiment_wrapper(experiment_names: str | list[str], must_exist: bool = False, strict_iterable: bool = True):

    if must_exist:
        with open("experiments.json", "r") as f:
            existing_names = json.load(f)

        if isinstance(experiment_names, list):
            tmp = experiment_names
        else:
            tmp = [experiment_names]

        for name in tmp:
            if name not in existing_names:
                raise DEVICE_EXCEPTION(f"YOU CALLED FOR '{name}', BUT NOBODY CAME")
                    
    #----------------------------------------------------------------------------
    
    if experiment_names == "all" or experiment_names == ["all"]:
        return list(all_experiments())
    else:
        xp = load_experiments(experiment_names, must_exist)
        if not strict_iterable or isinstance(xp, list):
            return xp
        else:
            return [xp]
        
#=============================================================================================================

def experiment_names_from_dir(dir, requested_experiments: str) -> list[str]:
    existing_names = os.listdir(dir)

    if requested_experiments == "all":
        requested_experiments = ",".join(existing_names)
    else:
        for name in requested_experiments.split(','):
            if name not in existing_names:
                raise DEVICE_EXCEPTION(f"YOU CALLED FOR '{name}', BUT NOBODY CAME")
            
    return [name for name in requested_experiments.split(",") if name in existing_names]
from ..helper import DEVICE_EXCEPTION
import json
from more_itertools import always_iterable

class ExperimentManager():
    '''
    A class for managing experiment templates
    '''
    experiments: dict

    CHOSEN_DOCS = {
        'pubmed': [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 1106]
    }

    DEFAULT_EXPERIMENT = {
        "title": "Default",
        "chaining_method": "iterative",
        "threshold": 0.6,
        "round_limit": None,
        "chain_pooling_method": "average",
        "n_components": 25,
        "min_dista": 0.1,
        "remove_duplicates": True,
        "normalize": True,
        "min_cluster_size": 3,
        "min_samples": 5,
        "n_neighbors": 15,
        "cluster_pooling_method": "average",
        "cluster_normalize": True
    }

    #---------------------------------------------------------------------------

    def __init__(self, path: str):
        '''
        A class for managing experiment templates

        Arguments
        ---
        path: str
            Path to the ```json``` file containing the experiment descriptions
        '''
        with open(path, "r") as f:
            self.experiments = json.load(f)
            self.experiments['default'] = self.DEFAULT_EXPERIMENT

    #---------------------------------------------------------------------------

    def document_index(self, index_name: str, doc_id: int, fallback: int = None) -> int:
        '''
        For a specific Elasticsearch index, return the position of a document in the list of test docs

        Arguments
        ---
        index_name: str
            Name of the Elasticsearch index
        doc_id: int
            The requested document
        fallback: int
            Value to return in case the document is not in the test list

        Returns
        ---
        index: int
            The index of the requested test document, or ```fallback``` if the document is not a test document
        '''
        try:
            return self.CHOSEN_DOCS.get(index_name, list(range(10))).index(doc_id)
        except ValueError:
            return fallback
        
    #---------------------------------------------------------------------------
        
    def select_experiments(self, experiment_names: str|list[str]|None = None, must_exist: bool = True, iterable: bool = False) -> dict | list[dict]:
        '''
        Return an experiment's parameters

        Arguments
        ---
        experiment_names: str | list[str] | None
            The names of the experiments to return. If set to ```None```, then all experiments are returned
        must_exist: bool
            Throw an exception if the experiment doesn't exist. Otherwise, return the default experiment. Defaults to ```True```
        iterable: bool
            If ```True```, then the returned value is always an iterable regardless of number of elements

        Returns
        ---
        params: dict | list[dict]
            Experiment parameters
        '''

        ret = None

        if isinstance(experiment_names, list) and len(experiment_names) == 1:
            experiment_names = experiment_names[0]

        #For a single requested experiment...
        if type(experiment_names) is str:
            
            if experiment_names in self.experiments:
                ret = self.DEFAULT_EXPERIMENT | self.experiments[experiment_names] | {'name': experiment_names}
            else:
                if must_exist:
                    raise DEVICE_EXCEPTION("THIS NEXT EXPERIMENT SEEMS... VACANT")
                else:
                    print(f"Could not find experiment '{experiment_names}'. Using default params")
                    ret = self.DEFAULT_EXPERIMENT | {'name': "default"}
        
        #For a list of requested experiments...
        elif isinstance(experiment_names, list):
            ret = [self.DEFAULT_EXPERIMENT | v | {'name': k} for k,v in self.experiments.items() if k in experiment_names]
        
        #Return all experiments
        elif experiment_names is None:
            ret = [self.DEFAULT_EXPERIMENT | v | {'name': k} for k,v in self.experiments.items()]

        if iterable:
            return list(always_iterable(ret, base_type=dict))
        else:
            return ret
        
    #---------------------------------------------------------------------------
        
    def experiment_names(self) -> set[str]:
        '''
        Return all experiment names
        '''
        return set(self.experiments)
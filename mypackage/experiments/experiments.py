from ..helper import DEVICE_EXCEPTION
import json
from more_itertools import always_iterable
 
from ..elastic import Session, ElasticDocument, ScrollingCorpus, NotFoundError
from ..query import Query

import re

class ExperimentManager():
    '''
    A class for managing experiment templates, as well as other preset values
    '''
    experiments: dict[str, dict]
    queries: list[dict]
    index_defaults: dict[str, dict]

    DEFAULT_EXPERIMENT = {
        "title": "Default",
        "sentence_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chaining_method": "iterative",
        "threshold": 0.6,
        "round_limit": None,
        "chain_pooling_method": "average",
        "min_chains": 28,
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
            data = json.load(f)
            self.experiments = data['experiments']
            self.experiments['default'] = self.DEFAULT_EXPERIMENT
            self.index_defaults = data['index_defaults']
            self.queries = data['queries']

    #---------------------------------------------------------------------------

    def get_docs(self, docs_list: str, sess: Session, *, scroll_batch_size: int=5000, scroll_limit: int = None, scroll_time: str="1000s") -> list[ElasticDocument]:
        #If docs are not specified, then a predefined set of docs is selected
        if not docs_list:
            docs_to_retrieve = self._get_docs_for_index(sess.index_name, list(range(10)))
            return [ElasticDocument(sess, doc, text_path="article") for doc in docs_to_retrieve]
        elif docs_list == "-1":
            return ScrollingCorpus(sess, batch_size=scroll_batch_size, limit=scroll_limit, scroll_time=scroll_time, doc_field="article")
        else:
            docs_to_retrieve = []
            for doc_set in docs_list.split(","):
                if res := re.match(r"^(?P<start>\d+)-(?P<end>\d+)$", doc_set):
                    docs_to_retrieve += list(range(int(res.group('start')), int(res.group('end'))+1))
                else:
                    docs_to_retrieve += [int(doc_set)]
            
            res = [ElasticDocument(sess, doc, text_path="article") for doc in docs_to_retrieve]

            '''
            if fetch:
                for i, d in enumerate(res):
                    try:
                        d.get()
                    except NotFoundError as e:
                        if skip_missing_docs:
                            res[i] = None
                        else:
                            raise e
            '''
                    
            return res
        


    #---------------------------------------------------------------------------

    def _get_docs_for_index(self, index_name: str, fallback = None) -> list[int]:
        temp = self.index_defaults.get(index_name, None)
        if temp is not None:
            return temp['docs']
        else:
            return fallback

    #---------------------------------------------------------------------------

    def _get_queries_for_index(self, index_name: str) -> list[Query]:
        return [
            Query(
                x['id'],
                x['query'],
                source = self.index_defaults[index_name]['source'],
                text_path = self.index_defaults[index_name]['text_path']
            )
            for x in self.queries
        ]
    
    #---------------------------------------------------------------------------

    def get_queries(self, query: str | None, index_name: str) -> list[Query]:
        #Get all queries
        if query is None:
            return self._get_queries_for_index(index_name)
        else:
            queries = {q.id: q for q in self._get_queries_for_index(index_name)}

            #For single IDs, we check if they are a valid id. If so, proceed
            #For multiple IDs, we check if it follows the regex, then we check each ID individually, potentially throwing error
            #If the ID is not valid on it's own AND doesn't match the multi-id pattern, we assume the query is dynamic
            if query in queries or re.match(r"^\d+(,\d+)+$", query):
                res = []
                for qid in query.split(","):
                    if qid in queries:
                        res.append(queries[qid])
                    else:
                        raise Exception(f"There is no query with ID '{qid}'")

                return res
            #Dynamic query
            else:
                index_defaults = self.index_defaults[index_name]
                return [Query(-1, query, source=index_defaults['source'], text_path=index_defaults['text_path'])]

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
            return self._get_docs_for_index(index_name, list(range(10))).index(doc_id)
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
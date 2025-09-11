from dataclasses import dataclass, field, fields
from typing import Literal, get_origin, get_args
from types import SimpleNamespace
import argparse
from collections import defaultdict
from rich.tree import Tree

#========================================================================================================================

@dataclass
class Arguments():
    '''
    This class determines dynamically both the client's command-line arguments and the arguments the server uses to control execution
    '''

    #Property names are the long forms
    #These are the actual names being used
    #To specify short-form, use metadata

    cluster: int = field(default=-1, metadata={"help": "The cluster number", "short": "c"})
    experiment: str = field(default="default", metadata={"help": "Experiment", "short": "x"})
    selection_method: Literal["topk", "thres"] = field(default="thres", metadata={"help": "Best cluster selection method", "short": "s"})
    print: bool = field(default=True, metadata={"help": "Print console messages"})
    stats: bool = field(default=False, metadata={"help": "Print cluster stats"})
    summ: bool = field(default=True, metadata={"help": "Summarize"})
    context_expansion_threshold: float = field(default=0.01, metadata={"help": "Context expansion threshold", "short": "cet"})
    cand_filter: float = field(default=-7, metadata={"help": "Initial candidate filter, to exclude very bad candidates from expansion"})
    candidate_sorting_method: str = field(default="flat_relevance", metadata={"help": "Candidate sorting method", "short": "csm"})
    eval: bool = field(default=False, metadata={"help": "Enable evaluation mode"})
    index: str = field(default="pubmed", metadata={"help": "Index name", "short": "i"})
    num_summaries: int = field(default=1, metadata={"help": "Number of summaries to generate from the same input", "short": "nsumm"})
    num_documents: int = field(default=10, metadata={"help": "Number of documents to retrieve from Elasticsearch", "short": "ndocs"})

    query: str = field(default=None, metadata={"help": "Numeric query ID or query string", "short": "q", "client_only": True})
    store_as: str = field(default=None, metadata={"help": "Filename to store summarization results in. Default is no store", "client_only": True})

    #-------------------------------------------------------------------------------------

    @classmethod
    def from_argparse(cls, args):
        '''
        Generate an Arguments object after filling in the values through the console
        '''
        args_dict = {name: getattr(args, name.replace("_", "-")) for name in map(lambda f: f.name, fields(Arguments))}
        return cls(**args_dict)
    
    #-------------------------------------------------------------------------------------

    #Fallback for short argument names, which don't actually exist as property names
    def __getattr__(self, attr):
        for f in fields(Arguments):
            if attr == f.metadata.get("short", None):
                return getattr(self, f.name)
        
        raise AttributeError(f"'Arguments' object has no attribute '{attr}'")
    
    #-------------------------------------------------------------------------------------
    
    @classmethod
    def setup_arguments(cls, parser: argparse.ArgumentParser):
        '''
        Uses the dataclass fields to create respective argparse arguments.
        We can then call parser.parse_args() on the resulting parser to get the commandline arguments

        Arguments
        ---
        parser: ArgumentParser
            The parser that will be populated with arguments
        '''

        for f in fields(Arguments):
            name = f.name.replace("_", "-")
            args = None

            #Common keyword arguments
            kwargs = {
                'help': f.metadata['help'],
                'default': f.default,
                'dest': name,
                'type': f.type,
                'action': 'store'
            }

            #Short and long argument names
            if (short_name := f.metadata.get('short', None)) is not None:
                args = (f"-{short_name}", f"--{name}")
            else:
                args = (f"--{name}",)

            #Boolean arguments
            #If the flag is true by default, create a complementary argument that disables it
            if f.type is bool:
                del kwargs['type'] #Flags must not have a type
                kwargs['action'] = 'store_true'

                if f.default == True:
                    group = parser.add_mutually_exclusive_group()
                    group.add_argument(*args, **kwargs)
                    group.add_argument(f"--no-{name}", action="store_false", default=True, dest=name, help=f.metadata['help'])
                else:
                    parser.add_argument(*args, **kwargs)
            else:
                #Arguments with choices
                if get_origin(f.type) is Literal:
                    kwargs['type'] = str
                    kwargs['choices'] = list(get_args(f.type))

                #Create the argument
                parser.add_argument(*args, **kwargs)

    #-------------------------------------------------------------------------------------

    def to_dict(self, ignore_defaults: bool = False, ignore_client_args: bool = False) -> dict:
        '''
        Get a dictionary representation of the commandline arguments
        '''
        data = None

        if ignore_defaults:
            data = {f.name: getattr(self, f.name) for f in fields(Arguments) if f.default != getattr(self, f.name) and not (f.metadata.get('client_only') and ignore_client_args)}
        else:
            data = {f.name: getattr(self, f.name) for f in fields(Arguments) if not (f.metadata.get('client_only') and ignore_client_args)}
    
        return data
    
    #-------------------------------------------------------------------------------------
    
    def to_namespace(self, ignore_defaults: bool = False) -> SimpleNamespace:
        return SimpleNamespace(**self.to_dict(ignore_defaults=ignore_defaults, ignore_client_args=True))
        
    #-------------------------------------------------------------------------------------

    def __rich__(self):
        return self.to_dict()
    
    #-------------------------------------------------------------------------------------

    def validate(self) -> tuple[bool, str]:
        if self.num_summaries < 1:
            return False, "Number of summaries must be greater than zero"
        
        return True, "ok"
    
#========================================================================================================================

def create_time_tree(times: dict , rename_map: dict = None):
    mytimes = defaultdict(float, {k:round(v, 3) for k,v in times.items()})
    result = {
        'elastic': mytimes['elastic'],
        'query_encode': mytimes['query_encode'],
        'cluster_retrieval': mytimes['cluster_retrieval']
    }

    tree = Tree(f"[green]Total time: [cyan]{sum(mytimes.values()):.3f}s[/cyan]")
    tree.add(f"[green]Elasticsearch time: [cyan]{mytimes['elastic']:.3f}s[/cyan]")
    tree.add(f"[green]Query encoding: [cyan]{mytimes['query_encode']:.3f}s[/cyan]")
    tree.add(f"[green]Cluster retrieval: [cyan]{mytimes['cluster_retrieval']:.3f}s[/cyan]")

    #Cross-score times
    temp = sum(v for k,v in mytimes.items() if k.startswith('cross_score'))
    result['cross_scores'] = temp
    score_tree = tree.add(f"[green]Cross-scores: [cyan]{temp:.3f}s[/cyan]")
    for k,v in mytimes.items():
        if k.startswith('cross_score'):
            score_tree.add(f"[green]Cluster {k[12:]}: [cyan]{v:.3f}s[/cyan]")

    #Context expansion times
    temp = sum(v for k,v in mytimes.items() if k.startswith('context_expansion'))
    result['context_expansion'] = temp
    context_tree = tree.add(f"[green]Context expansion: [cyan]{temp:.3f}s[/cyan]")
    for k,v in mytimes.items():
        if k.startswith('context_expansion'):
            context_tree.add(f"[green]Cluster {k[18:]}: [cyan]{v:.3f}s[/cyan]")

    summary_tree = tree.add(f"[green]Summarization[/green]: [cyan]{mytimes['summary_time']}s[/cyan]")
    summary_tree.add(f"[green]Response time[/green]: [cyan]{mytimes['summary_response_time']:.3f}s[/cyan]")

    result['summary_time'] = mytimes['summary_time']
    result['summary_response_time'] = mytimes['summary_response_time']
    
    result['total'] = sum(v for v in result.values())

    if rename_map:
        result = {rename_map.get(k, k): v for k, v in result.items()}

    return tree, result
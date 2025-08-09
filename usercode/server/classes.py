from mypackage.helper import NpEncoder

from dataclasses import dataclass, field, fields
from typing import Literal, get_origin, get_args, Any
import json
import argparse

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
    candidate_sorting_method: str = field(default="flat_relevance", metadata={"help": "Candidate sorting method", "short": "csm"})

    test: int = field(default=0, metadata={'help': 'Test', 'client_only': True})

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

    def __rich__(self):
        return self.to_dict()

#========================================================================================================================

@dataclass
class Message():
    type: str
    contents: Any = field(default=None)

    @classmethod
    def from_sse_event(cls, event):
        try:
            obj = json.loads(event.data)
            return cls(obj['type'], obj['contents'])
        except:
            print(event)

    def to_json(self, string=False) -> dict:
        data = {'type': self.type, 'contents': self.contents}
        if string:
            return json.dumps(data, cls=NpEncoder, ensure_ascii=False)
        return data

    def to_sse(self):
        return f"data: {self.to_json(string=True)}\n\n".encode("utf-8")
    
    
#========================================================================================================================
from mypackage.helper import NpEncoder

from dataclasses import dataclass, field, fields
from typing import Literal, get_origin, get_args, Any
import json
import argparse

#========================================================================================================================

@dataclass
class Arguments():
    '''
    This class determines dynamically both the command-line arguments and the query parameters the request can take
    '''
    c: int = field(default=-1, metadata={"help": "The cluster number"})
    x: str = field(default="default", metadata={"help": "Experiment"})
    s: Literal["topk", "thres"] = field(default="thres", metadata={"help": "Best cluster selection method"})
    print: bool = field(default=True, metadata={"help": "Print console messages"})
    stats: bool = field(default=False, metadata={"help": "Print cluster stats"})
    summ: bool = field(default=True, metadata={"help": "Summarize"})
    cet: float = field(default=0.01, metadata={"help": "Context expansion threshold"})
    csm: str = field(default="flat_relevance", metadata={"help": "Candidate sorting method"})

    #-------------------------------------------------------------------------------------

    @classmethod
    def from_argparse(cls, args):
        args_dict = {f.name: getattr(args, f.name) for f in fields(Arguments)}
        return cls(**args_dict)
    
    #-------------------------------------------------------------------------------------
    
    @classmethod
    def from_query_params(cls, args):
        args_dict = {}
        for f in fields(Arguments):
            dtype = (str if hasattr(f.type, '__origin__') and f.type.__origin__ is Literal else f.type)
            value = args.get(f.name, f.default)
            if dtype.__name__ == "bool" and type(value) is str:
                value = (value.lower() == "true")
            args_dict = {**args_dict, f.name: dtype(value)}
        return cls(**args_dict)
    
    #-------------------------------------------------------------------------------------
    
    @classmethod
    def setup_arguments(cls, parser: argparse.ArgumentParser):

        for f in fields(Arguments):
            if f.type is bool:
                parser.add_argument(f"--{f.name}", action="store_true", dest=f.name, default=f.default, help=f.metadata['help'])
                if f.default == True:
                    parser.add_argument(f"--no-{f.name}", action="store_false", default=True, dest=f.name, help=f.metadata['help'])
            elif get_origin(f.type) is Literal:
                parser.add_argument(f"-{f.name}", action="store", type=str, default=f.default, help=f.metadata['help'], choices=list(get_args(f.type)))
            else:
                parser.add_argument(f"-{f.name}", action="store", type=f.type, default=f.default, help=f.metadata['help'])

    #-------------------------------------------------------------------------------------

    def get_dict(self, ignore_defaults: bool = False) -> dict:
        if ignore_defaults:
            return {f.name: getattr(self, f.name) for f in fields(Arguments) if f.default != getattr(self, f.name)}
        else:
            return {f.name: getattr(self, f.name) for f in fields(Arguments)}
        
    #-------------------------------------------------------------------------------------

    def __rich__(self):
        return self.get_dict()

#========================================================================================================================

@dataclass
class Message():
    type: str
    contents: Any

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
from dataclasses import dataclass, field, fields
from typing import Literal, get_origin, get_args, Any
import json
from mypackage.helper import NpEncoder

@dataclass
class Arguments():
    c: int = field(default=0, metadata={"help": "The cluster number"})
    s: Literal["topk", "thres"] = field(default="thres", metadata={"help": "Best cluster selection method"})
    print: bool = field(default=False, metadata={"help": "Print the cluster's text"})
    stats: bool = field(default=False, metadata={"help": "Print cluster stats"})
    summ: bool = field(default=True, metadata={"help": "Summarize"})
    cet: float = field(default=0.01, metadata={"help": "Context expansion threshold"})
    csm: str = field(default="flat_relevance", metadata={"help": "Candidate sorting method"})

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
            return json.dumps(data, cls=NpEncoder)
        return data

    def to_sse(self):
        return f"data: {self.to_json(string=True)}\n\n".encode("utf-8")
    
    
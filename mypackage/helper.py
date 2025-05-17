from rich.panel import Panel
from rich.console import Console
from rich.table import Table
import sys
from itertools import islice
from functools import wraps
import json
import numpy as np

console = Console()

#===================================================================================

def panel_print(text: str, title: str = ""):
    console.print(Panel(text, title=title, title_align="left", border_style="bold cyan"))

#===================================================================================

def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider

#===================================================================================

def total_size(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, list):
        size += sum(total_size(item) for item in obj)
    return size / 1024 // 1024

#===================================================================================

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

#===================================================================================

def file_batch(fname: str, batch_size: int):
    byte_offset = 0
    current_lines = 0
    
    yield 0

    with open(fname, "r") as f:
        for line in f:
            byte_offset += len(line.encode('utf-8'))

            if current_lines == batch_size - 1:
                yield byte_offset
                current_lines = 0
            else:
                current_lines += 1

#===================================================================================

def line_count(file: str):
    with open(file, "r") as f:
        line_count = sum(1 for _ in f)

    return line_count

#===================================================================================

def lock_kwargs(func, **locked_kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.update(locked_kwargs)
        return func(*args, **kwargs)
    return wrapper

#===================================================================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

#===================================================================================

def create_table(column_names: list[str], data: dict, *, title:str|None =None, round=True) -> Table:
    table = Table(title=title, title_justify="left", header_style="")

    if round:
        for key in data:
            if not isinstance(data[key], list):
                data[key] = [data[key]]

            for i in range(len(data[key])):
                if isinstance(data[key][i], (float, np.double, np.float32, np.float64)):
                    data[key][i] = f"{np.round(data[key][i], decimals=3):.3f}"
                else:
                    data[key][i] = f"{data[key][i]}"

    for i, name in enumerate(column_names):
        if i == 0:
            table.add_column(name, style="")
        else:
            table.add_column(name)

    for name, value_list in data.items():
        table.add_row(name, *value_list)

    return table
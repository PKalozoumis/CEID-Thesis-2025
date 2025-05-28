from rich.panel import Panel
from rich.console import Console, Group
from rich.table import Table
import sys
from itertools import islice
from functools import wraps
import json
import numpy as np
import copy

console = Console()

#=============================================================================================================

class DEVICE_EXCEPTION(Exception):
    pass

#===================================================================================

def panel_print(text: str = "", title: str = "", *, return_panel=False, expand: bool = True):
    if isinstance(text, list):
        panel = Panel(Group(*text), title=title, title_align="left", border_style="bold cyan", expand=expand)
        if return_panel:
            return panel
        console.print(panel)
    else:
        panel = Panel(text, title=title, title_align="left", border_style="bold cyan", expand=expand)
        if return_panel:
            return panel
        console.print(panel)

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

def round_data(data: list, to_string = False):
    data = copy.deepcopy(data)

    for i in range(len(data)):
        if isinstance(data[i], (float, np.double, np.float32, np.float64)):
            data[i] = f"{np.round(data[i], decimals=3):.3f}" if to_string else np.round(data[i], decimals=3)
        else:
            data[i] = f"{data[i]}" if to_string else data[i]
        
    return data

#===================================================================================

def create_table(column_names: list[str], data: dict, *, title:str|None = None, round=True) -> Table:
    table = Table(title=title, title_justify="left", header_style="")
    data = copy.deepcopy(data)
    
    for key in data:
        if not isinstance(data[key], list):
            data[key] = [data[key]]

        if round:
            data[key] = round_data(data[key], to_string=True)
        else:
            for i in range(len(data[key])):
                data[key][i] = f"{data[key][i]}"

    for i, name in enumerate(column_names):
        if i == 0:
            table.add_column(name, style="")
        else:
            table.add_column(name)

    for name, value_list in data.items():
        table.add_row(name, *value_list)

    return table

#===================================================================================

def write_to_excel_tab(worksheet, title: str, row_data: dict[str, list], column_names, *, row_offset: int|None = None, column_offset: int|None = None, name_fmt, title_fmt, global_fmt, first_width: float = 0) -> int | tuple[int, float]:
    '''
    Returns
    ---
    new_offset: int
        The new position where the next table should be written

    first_width: float
        Maximum width of the first column. Only if row_offset is not None, meaning that we draw tables vertically
    '''

    if row_offset is None and column_offset is None:
        raise ValueError("Set either row_offset or column_offset")
    
    if row_offset is None:
        temp_row_offset = 0
    elif column_offset is not None:
        raise ValueError("row_offset and column_offset cannot both be set")
    else:
        temp_row_offset = row_offset

    if column_offset is None:
        temp_col_offset = 0
    elif row_offset is not None:
        raise ValueError("row_offset and column_offset cannot both be set")
    else:
        temp_col_offset = column_offset

    #Title
    worksheet.merge_range(temp_row_offset, temp_col_offset, temp_row_offset, temp_col_offset + len(column_names), title, title_fmt)

    #Corner element
    worksheet.write(temp_row_offset+1, temp_col_offset, "", name_fmt)

    #Column names and widths
    for colnum, colname in enumerate(column_names):
        worksheet.write(temp_row_offset+1, temp_col_offset+colnum+1, colname, name_fmt)
        worksheet.set_column(temp_col_offset+colnum+1, temp_col_offset+colnum+1, max(8.43, len(colname)))

    #Row names and data
    max_rowname = max(0, first_width)
    for rownum, (rowname, rowlist) in enumerate(row_data.items()):
        max_rowname = max(max_rowname, len(rowname))
        worksheet.write(temp_row_offset+1+rownum+1, temp_col_offset, rowname, name_fmt)
        worksheet.write_row(temp_row_offset+1+rownum+1, temp_col_offset+1, round_data(rowlist, to_string=True), global_fmt)

    worksheet.set_column(temp_col_offset,temp_col_offset,max_rowname)

    if row_offset is None:
        return temp_col_offset + len(column_names) + 2
    else:
        return temp_row_offset + 1 + len(row_data) + 2, max_rowname

#===================================================================================

def binary_search_ranges(ranges: list[tuple], target: int):
    low, high = 0, len(ranges) - 1

    while low <= high:
        mid = (low + high) // 2
        start, end = ranges[mid]

        if start <= target <= end:
            return mid  # target inside this range
        elif target < start:
            high = mid - 1
        else:  # target > end
            low = mid + 1

    return -1  # not found in any range
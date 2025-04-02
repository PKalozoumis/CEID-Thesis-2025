from rich.panel import Panel
from rich.console import Console

console = Console()

def panel_print(text: str, title: str = ""):
    console.print(Panel(text, title=title, title_align="left", border_style="bold cyan"))

def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider
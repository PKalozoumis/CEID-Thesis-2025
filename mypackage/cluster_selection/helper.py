from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .classes import SelectedCluster, SummaryCandidate
    
from mypackage.helper import panel_print, rich_console_text
from rich.rule import Rule
from rich.console import Console

#========================================================================================================

def print_candidate_states(candidate: SummaryCandidate):
    panel_lines = []
    
    for i, state in enumerate(candidate.history):
        line = ""
        if i == candidate.selected_state:
            line += "[cyan]-> [/cyan]"
        line += f"[green]{i:02}.[/green] "
        line += f"Chain [green]{state.first_index}[/green] " if len(state) == 1 else f"Chains [green]{state.first_index}-{state.last_index}[/green] "
        line += f"with score [cyan]{round(state.score, 3):.3f}[/cyan], improvement score [cyan]{state.improvement_score:.3f}[/cyan])"
        line += f" ({' -> '.join(state.actions)})".ljust(19)
        panel_lines.append(line)

    panel_print(panel_lines, title= f"States of candidate {candidate.id}", expand=False)

#========================================================================================================

def print_candidates(focused_cluster: SelectedCluster, *, print_action: bool = False, current_state_only: bool = False, title: str | None = None, return_text: bool = False):

    panel_lines = []

    for i, c in enumerate(focused_cluster.candidates):
        line_text = f"[green]{i:02}[/green]. "
        line_text_list = []

        col = "green" if c.expandable else "red"

        #This will crash if one candidate has more history entries than the rest
        #whatever
        for num, state in enumerate([c.context] if current_state_only else c.history):
            #Yes I know this is goofy
            big_chain_in_column = False
            for c1 in focused_cluster.candidates:
                if current_state_only and len(c1.context) > 1:
                    big_chain_in_column = True
                    break
                elif not current_state_only and num < len(c1.history) and len(c1.history[num]) > 1:
                    big_chain_in_column = True
                    break

            if not big_chain_in_column:
                temp = f"[{col}]{state.chains[0].index:03}[/{col}]"
            else:   
                temp = f"[{col}]{state.chains[0].index:03}[/{col}]".rjust(19).ljust(23 if c.expandable else 19) if len(state) == 1 else f"[{col}]{state.id}[/{col}]"

            history_text = f"Chain {temp}" if len(state) == 1 else f"Chains {temp}"
            history_text += f" with score " + f"[cyan]{state.score:.3f}[/cyan]".rjust(20)
            history_text += f" ({' -> '.join(state.actions)})".ljust(19) if print_action else ""
            line_text_list.append(history_text)

        line_text += " [red]->[/red] ".join(line_text_list)
        panel_lines.append(line_text)

    panel_lines.append(Rule())

    #Overall cluster score
    if current_state_only:
        panel_lines.append(f"Cluster score: [cyan]{focused_cluster.cross_score:.3f}[/cyan]")
    else:
        cluster_scores = [
            f"[cyan]{focused_cluster.historic_cross_score(i):.3f}[/cyan]" for i in range(len(focused_cluster.candidates[0].history))
        ]
        panel_lines.append(f"Cluster score: " + " [red]->[/red] ".join(cluster_scores))

    panel = panel_print(panel_lines, title= title or f"For cluster {focused_cluster.id if focused_cluster is not None else None}", expand=False, return_panel=return_text)
    
    if return_text:
        return rich_console_text(panel)
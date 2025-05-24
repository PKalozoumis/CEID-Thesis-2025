#I want to see if next/previous sentence retrieval works

import sys
import os
sys.path.append(os.path.abspath("../.."))

from mypackage.helper import panel_print
from mypackage.elastic import Session, ElasticDocument
from mypackage.storage import load_pickles
from rich.console import Console
from rich.rule import Rule
from rich.padding import Padding

sess = Session("pubmed-index", cache_dir="../cache", use="cache")
doc = ElasticDocument(sess, 1923, text_path="article")
p = load_pickles(sess, "../experiments/pubmed-index/pickles/default", doc)

console = Console()

#console.print(p.doc.text)

N = 20
last_sentence = None

#Forward
#-----------------------------------------------------------
s = p.chains[0].sentences[0]
group = []
for n in range(1,N):
    next_sentences = s.next(n, force_list=True)
    group.extend([
        s.text + "\n\n" + ("\n\n".join([temp.text for temp in next_sentences])),
        Padding(Rule(), (0, 0, 1, 0))
    ])

    last_sentence = next_sentences[-1]

panel_print(group)
print("\n\n\n\n\n")

#Now in reverse....
#-----------------------------------------------------------
s = last_sentence
group = []
for n in range(1,N):
    next_sentences = s.prev(n, force_list=True)
    group.extend([
        ("\n\n".join([temp.text for temp in next_sentences])) + "\n\n" + s.text,
        Padding(Rule(), (0, 0, 1, 0))
    ])

panel_print(group)
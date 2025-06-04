import numpy as np

def split_to_sentences(text: str, *, sep: str | None = "\n") -> list[str]:
    sentences = text.split(sep)
    if sentences[-1] == '':
        sentences = sentences[:-1]

    return sentences
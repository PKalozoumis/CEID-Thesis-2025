import json
import os
from huggingface_hub import try_to_load_from_cache
from nltk.tokenize import sent_tokenize

def split_to_sentences(text: str, *, sep: str | None = None) -> list[str]:
    '''
    Splits a piece of text into sentences

    Arguments
    ---
    text: str
        The piece of text to split
    sep: str | None
        The separator used for splitting. Defaults to ```None```, meaning that ```nltk.tokenize.sent_tokenize``` is
        used to automatically detect sentence boundaries

    Returns
    ---
    sentences: list[str]
        A list of the text's sentences
    '''
    if sep is None:
        sentences = sent_tokenize(text)
    else:
        sentences = text.split(sep)
        if sentences[-1] == '':
            sentences = sentences[:-1]

    return sentences

def sentence_transformer_from_alias(model_name_or_alias: str, alias_file: str = "model_aliases.json") -> str:
    name = None
    if not os.path.isfile(alias_file):
        name = model_name_or_alias
    else:
        with open(alias_file, 'r') as f:
            data = json.load(f)
            name = data.get(model_name_or_alias, model_name_or_alias)
    
    if try_to_load_from_cache(name, "config.json") is not None:
        return name
    else:
        raise Exception(f"Model {name} is not downloaded")
    

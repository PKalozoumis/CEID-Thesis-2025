import json
import os
from huggingface_hub import try_to_load_from_cache

def split_to_sentences(text: str, *, sep: str | None = "\n") -> list[str]:
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
    

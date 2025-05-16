import numpy as np

def split_to_sentences(text) -> list[str]:
    sentences = text.split("\n")
    if sentences[-1] == '':
        sentences = sentences[:-1]

    return sentences

def cosine_sim(vec1, vec2) -> float:
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
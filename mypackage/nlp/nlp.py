import spacy

nlp = spacy.load("en_core_web_lg")
doc = nlp("Apple was founded by Steve Jobs in California.")

for ent in doc.ents:
    print(ent.text, ent.label_)
import sys
import os

sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import ElasticDocument, Session
import os
import json
import lmstudio as lms
from lmstudio import LMStudioClientError
import platform
import netifaces

#Initialize model
#=================================================================================================
if 'microsoft' in platform.uname().release.lower():
    gateway = netifaces.gateways()['default'][netifaces.AF_INET][0]
    api_host = f"{gateway}:1234"
else:
    api_host = "localhost:1234"


try:
    lms.get_default_client(api_host)
    model = lms.llm("meta-llama-3.1-8b-instruct")
except LMStudioClientError:
    #print("reusing client")
    model = lms.llm("meta-llama-3.1-8b-instruct")

#======================================================================================================

def generate_query(text: str):
    '''
    Generates a query from the given document

    Arguments
    ---
    text: str
        The document
    '''

    system_prompt = f'''You are a helpful assistant that generates queries. Your task is to generate **two relevant search queries** for each document summary provided. 
These queries simulate what a user might type into a search system to find this document.

## Guidelines:
1. The first query should be a **short, simple search phrase**, not a question.
2. The second query should be phrased as a **question**.
3. Queries must be concise and easy to understand.
4. Assume the user has no prior knowledge of the document.
5. Avoid overly specific details; focus on maximizing **general** relevance.
'''

    schema = {
        "type": "object",
        "properties": {
            "q1": { "type": "string" },
            "q2": { "type": "string" },
        },
        "required": ["q1", "q2"],
    }
    
    chat = lms.Chat()
    
    chat.add_system_prompt(system_prompt)
    chat.add_user_message(text)
    result = model.respond(chat, response_format=schema)

    return json.loads(result.content)

#======================================================================================================

if __name__ == "__main__":
    sess = Session("pubmed", base_path="../common")
    docs = (ElasticDocument(sess, i, text_path="summary") for i in range(1014,6000))

    out_file = open("queries.txt", "a")
    out_file.write("\n")
    
    for doc in docs:
        try:
            print(f"For document {doc.id:04}\n{'='*30}")
            res = generate_query(doc.text)
            print(res, end="\n\n")
            out_file.write(json.dumps({'doc': doc.id, **res}) + "\n")
            out_file.flush()
        except Exception as e:
            print(e)
            continue

    out_file.close()
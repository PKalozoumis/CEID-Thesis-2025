import lmstudio as lms
import platform
import netifaces
import json

#Initialize model
#=================================================================================================
#Check if I'm using WSL
#If so, LMStudio is running the windows host, which is the gateway
if 'microsoft' in platform.uname().release.lower():
    gateway = netifaces.gateways()['default'][netifaces.AF_INET][0]
    api_host = f"{gateway}:1234"
else:
    api_host = "localhost:1234"


lms.get_default_client(api_host)
model = lms.llm("llama-3.2-3b-instruct")
#model = lms.llm("llama-3.2-1b-instruct")


#================================================================================================

def generate_query(text: str):
    '''
    Generates a query from the given document

    Arguments
    ---
    text: str
        The document
    '''

    system_prompt = f'''Your goal is to generate 2 relevant search queries for every document summary the user provides.
These queries are meant to represent the input of a retrieval system that will consider the document as relevant.
Each document will be provided as a separate user message.
The first query must represent a simple real user search query, that isn't formatted as a question
The second query will appear more like a question
You must strictly adhere to these guidelines:
1. Assume that the user has no knowledge of the given document and is trying to retrieve it. Avoid mentioning very specific details
2. Neither query must not be too long. Keep them short and concise
3. Avoid copying words directly from the source text. Utilize a balance between original and copied words
'''
    
    schema = {
        "type": "object",
        "properties": {
            "q1": { "type": "string" },
            "q2": { "type": "string" },
        },
        "required": ["q1", "q2"],
    }
    
    chat = lms.Chat(system_prompt)
    
    chat.add_user_message(text)
    result = model.respond(chat, response_format=schema)

    return json.loads(result.content)

#================================================================================================

def multi_summary_query(summaries: list[str]):
    '''
    Generates a query from the given document

    Arguments
    ---
    text: str
        The document
    '''

    system_prompt = f'''You will be provided a couple of summaries separated by lines (---).
Your goal is to generate two queries that would simultaneously retrieve both documents in a retrieval system.
You must identify parts of the documents that are common and try to formulate a question whose answer can be found by combining the partial documents
You must strictly adhere to these guidelines:
1. Assume that the user has no knowledge of the given documents and is trying to retrieve them. Avoid mentioning very specific details
2. The queries must not be too long. Keep it short and concise
3. Avoid copying words directly from the source text. Utilize a balance between original and copied words
4. Both queries must be able to retrieve both documents
'''
    
    schema = {
        "type": "object",
        "properties": {
            "q1": { "type": "string" },
            "q2": { "type": "string" },
        },
        "required": ["q1", "q2"],
    }
    
    chat = lms.Chat(system_prompt)
    
    chat.add_user_message("\n---\n".join(summaries))
    result = model.respond(chat, response_format=schema)

    return json.loads(result.content)
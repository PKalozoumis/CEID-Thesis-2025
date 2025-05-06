import lmstudio as lms
import platform
import netifaces
import json

schema = {
    "type": "object",
    "properties": {
        "q1": { "type": "string" },
        "q2": { "type": "string" },
    },
    "required": ["q1", "q2"],
}

#Check if I'm using WSL
#If so, LMStudio is running the windows host, which is the gateway
if 'microsoft' in platform.uname().release.lower():
    gateway = netifaces.gateways()['default'][netifaces.AF_INET][0]
    api_host = f"{gateway}:1234"
else:
    api_host = "localhost:1234"

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
    chat = lms.Chat(system_prompt)
    chat.add_user_message(text)
    result = model.respond(chat, response_format=schema)

    return json.loads(result.content)
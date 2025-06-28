from __future__ import annotations

import lmstudio as lms
from lmstudio import LMStudioClientError
import platform
import netifaces
import re

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .summarization.classes import SummarySegment

#================================================================================================

class LLMSession():
    model_name: str
    api_host: str
    model: lms.LLM

    def __init__(self, model_name: str = "llama-3.2-3b-instruct"):
        self.model_name = model_name

        #Check if I'm using WSL
        #If so, LMStudio is running the windows host, which is the gateway
        if 'microsoft' in platform.uname().release.lower():
            gateway = netifaces.gateways()['default'][netifaces.AF_INET][0]
            self.api_host = f"{gateway}:1234"
        else:
            self.api_host = "localhost:1234"

        client = lms.Client(api_host=self.api_host)
        self.model = client.llm.model(model_name)

#================================================================================================

def merge_summaries(llm: LLMSession, segments: list[SummarySegment], query: str):

    system_prompt = f'''Rewrite the following document into a coherent, fluent document (with paragraph breaks as needed) that focused on answering the following query: \"{query}\".
Only use content from the document. Do not paraphrase, interpret, or add new information, except to improve fluency. Keep original wording and phrasing. Ensure the result flows naturally and clearly.
Use all relevant information from the document that answers the query.
I will now provide the document:
'''
    
    schema = {
        "type": "object",
        "properties": {
            "answer": { "type": "string" },
        },
        "required": ["answer"],
    }
    
    chat = lms.Chat(system_prompt)
    text = "".join([seg.summary for seg in segments if len(seg.summary) > 0])
    
    chat.add_user_message(text)
    for fragment in llm.model.respond_stream(chat):
        yield fragment.content

    #return json.loads(result.content)

#================================================================================================

def llm_summarize(llm: LLMSession, query: str, text: str):

    system_prompt = f'''You are an expert summarizer. Given a query and a set of numbered documents separated by "---",
write a detailed, comprehensive summary that fully answers the query using only the information from the documents.
Every key fact or claim must include a clear citation referencing the document ID in square brackets (e.g., [doc1]).

- Integrate all relevant points and nuances from all documents.
- Do not add any information that is not explicitly stated in the documents.
- Structure the summary with multiple paragraphs if needed for clarity and depth.
- Alwaysnclude citations immediately after the facts or claims they support.

Query:
---
{query}

{text}
'''
    
    schema = {
        "type": "object",
        "properties": {
            "summary": { "type": "string" },
        },
        "required": ["summary"],
    }
    
    chat = lms.Chat(system_prompt)
    
    chat.add_user_message(text)

    temp_text = ""
    removed_json = False
    temp_temp = ""

    for fragment in llm.model.respond_stream(chat):
        temp_text += fragment.content

        #Clean up the json
        #-------------------------------------------------
        
        if removed_json:
            if res := re.search(r"\s?\"\s?$", temp_text):
                temp_temp = res.group()
                continue
            if res := re.search(r"\s?\"\s?}\s?$", temp_text):
                temp_text = temp_text[:-len(res.group())]
            else:
                temp_text += temp_temp
                temp_temp = ""

        if not removed_json:
            m = re.match(r"\s*{\s*\"\s?summary\s?\"\s?:\s?\"\s?", temp_text)
            if m:
                yield temp_text[len(m.group()):]
                temp_text = ""
                removed_json = True

        #Json is removed from output
        if removed_json:
            yield temp_text
            temp_text = ""
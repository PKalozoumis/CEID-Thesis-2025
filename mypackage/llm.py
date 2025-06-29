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
    '''
    A session for an LLM running in LM Studio
    '''
    model_name: str
    api_host: str
    model: lms.LLM

    def __init__(self, model_name: str = "meta-llama-3.1-8b-instruct"):
        '''
        A session for an LLM running in LM Studio

        Arguments
        ---
        model_name: str
            The name of the model
        '''
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

def llm_summarize(llm: LLMSession, query: str, text: str):

    system_prompt = f'''You are an expert summarizer. Given a query and a set of documents separated by "---",
write a detailed, comprehensive summary that fully answers the query using only the information from the documents.
All sentences of the summary must include a clear citation to the document's numeric ID in square brackets
where that fact or claim originates. The citation must only include a number.

- Integrate all relevant points and nuances from all documents.
- Do not add any information that is not explicitly stated in the documents.
- Structure the summary with multiple paragraphs if needed for clarity and depth.
'''
    
    system_prompt = f'''You are an expert summarizer. Given a query and a series of facts,
write a detailed, comprehensive summary that fully answers the query using only the information from the facts.
Each fact begins with an ID in the format <1234_0-5>. All sentences of the summary must include a clear citation to the fact's ID
where that claim originates, in the same format <1234_0-5>, with angle brackets (`<` and `>`) — not parentheses or any other symbol.

- Integrate all relevant points and nuances from all facts.
- Do not add any information that is not explicitly stated in the facts.
- Structure the summary with multiple paragraphs if needed for clarity and depth.
'''
    
    schema = {
        "type": "object",
        "properties": {
            "summary": { "type": "string" },
        },
        "required": ["summary"],
    }
    
    chat = lms.Chat(system_prompt)
    
    chat.add_user_message(f"Query: \"{query}\"\n{text}\nSummary:\n")

    temp_text = ""
    removed_json = False
    temp_temp = ""

    for fragment in llm.model.respond_stream(chat):
        if False:
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
        else:
            yield fragment.content
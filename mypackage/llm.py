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

def llm_summarize(llm: LLMSession, query: str, text: str, stop_dict):
    
    prompt = f'''You are a summarization expert. Given a query and a series of facts, you write a detailed, comprehensive summary that fully answers the query using only information from the facts.

### What is a fact:
A fact is a span of text that comes from a specific document and is relevant to the query.
Each fact always begins with an ID that looks like <1234_0-5>, followed by a colon.

### Instructions on how to generate the summary:
- Integrate all relevant points and nuances from all facts, but condensed
- Never add any information that is not explicitly stated in the facts.
- Structure the summary with multiple paragraphs for clarity and depth, but keep it concise

### Each sentence in the summary is accompanied by a citation to the relevant fact:
- The citation must always be at the end of the sentence
- The citation must always be in the same format <1234_0-5>, with just angle brackets (`<` and `>`).
- Never use parentheses or any other symbol for citations.
- Never surround the citations with parentheses.
- If a sentence is the result of two or more facts, then include citations for all facts, one after the other (e.g. <1234_0-1><5678_5-6>)

### This is an example of the desired output for one of the summary sentences:
Query: "What is the capital city of France?"  
Facts:
<1001_0-1>: Paris is the capital city of France and a major cultural center.
Summary:  
Paris is the capital city of France <1001_0-1>.

### Here is another example, where a summary sentence is supported by two or more facts:
Query: "What is notable about the current state of the Amazon rainforest?" 
Facts:
<1234_0-1>: The Amazon rainforest contains over 390 billion individual trees.
<5678_5-6>: In the last decade, deforestation in the Amazon has risen by 85%.
Summary:  
The Amazon rainforest contains over 390 billion trees, yet deforestation has increased by 85% in the last decade <1234_0-1><5678_5-6>.

### Now summarize the following:
Query: "{query}"
Facts:
{text}
Summary:

'''

    chat = lms.Chat()
    
    chat.add_user_message(prompt)

    temp_text = ""
    removed_json = False
    temp_temp = ""

    stream = llm.model.respond_stream(chat)
    for fragment in stream:
        if stop_dict['force_stop']:
            stream.cancel()
            stream.close()
            stop_dict['stopped'] = True
            yield stream, "."
        else:
            yield stream, fragment.content
        
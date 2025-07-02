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
    
    system_prompt = f'''You are an expert summarizer. Given a query and a series of facts,
write a detailed, comprehensive summary that fully answers the query using only the information from the facts.
Each fact begins with an ID in the format <1234_0-5>. At the end of each sentence in the summary, you must include a clear citation to the fact's ID
where that claim originates. A citation must strictly be in the same format <1234_0-5>, with just angle brackets (`<` and `>`).
Do not use parentheses or any other symbol for citations.
Do not surround the citation with parentheses.
If multiple citations are needed for the same sentence, keep them as two separate, consecutive references e.g. <1234_0><1234_1>

- Integrate all relevant points and nuances from all facts, but condensed
- Do not add any information that is not explicitly stated in the facts.
- Structure the summary with multiple paragraphs if needed for clarity and depth, but keep it concise
'''

    chat = lms.Chat(system_prompt)
    
    chat.add_user_message(f"Query: \"{query}\"\n{text}\nSummary:\n")

    temp_text = ""
    removed_json = False
    temp_temp = ""

    stream = llm.model.respond_stream(chat)
    for fragment in stream:
        if stop_dict['stop']:
            stream.cancel()
            stream.close()
            stop_dict['stopped'] = True
            yield stream, "amogus"
        else:
            yield stream, fragment.content
        
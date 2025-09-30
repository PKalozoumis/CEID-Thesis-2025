from __future__ import annotations

import requests
import json
from abc import ABC, abstractmethod
import lmstudio as lms
from lmstudio import LlmPredictionConfig
import platform
import netifaces
import re
from typing import Literal, Any, Generator

#================================================================================================

class LLMSession(ABC):
    model_name: str
    api_host: str
    cache_prompt: bool
    connection_obj: Any

    system_prompt = f'''You are a summarization expert. Given a query and a series of facts, you write a detailed, comprehensive summary that fully answers the query using only information from the facts.

### What is a fact:
A fact is a span of text that comes from a specific document and is relevant to the query.
Each fact always begins with an ID that looks like <1234_0-5>, followed by a colon.

### Instructions on how to generate the summary:
- Never repeat the query at the start. Only provide the summary and nothing else.
- Integrate all relevant points and nuances from all facts, but condensed
- Never add any information that is not explicitly stated in the facts.
- Structure the summary with multiple paragraphs for clarity and depth, but keep it concise

### Each sentence in the summary must be accompanied by a citation to the relevant fact:
- The citation must always be at the end of the sentence, followed by a fullstop.
- The citation must always be in the ID format <1234_0-5>, with just angle brackets (`<` and `>`).
- If a sentence uses multiple facts, list IDs back to back with no text in between (e.g. <1234_0-1><5678_5-6>).
- Never modify a fact's ID. Copy it as is.

### This is an example of the desired output for one of the summary sentences:
Query: "What is the capital city of France?"  
Facts:
<1001_0-1>: Paris is the capital city of France and a major cultural center.
Summary:  
Paris is the capital city of France <1001_0-1>.

### Here is another example, where a summary sentence is supported by two or more facts:
Query: "What is notable about the current state of the Amazon rainforest?" 
Facts:
<1234_0-0>: The Amazon rainforest contains over 390 billion individual trees.
<5678_5-6>: In the last decade, deforestation in the Amazon has risen by 85%.
Summary:  
The Amazon rainforest contains over 390 billion trees, yet deforestation has increased by 85% in the last decade <1234_0-0><5678_5-6>.
'''
    @classmethod
    def create(cls, backend: Literal["llamacpp", "lmstudio"], model_name: str = "meta-llama-3.1-8b-instruct", api_host=None) -> LLMSession:
        '''
        Create a session based on the selected backend
        '''
        if backend == "llamacpp":
            return LlamaCppSession(model_name, api_host)
        elif backend == "lmstudio":
            return LMStudioSession(model_name, api_host)

    @abstractmethod
    def summarize(self, query: str, text: str, stop_dict, *, cache_prompt: bool = False) -> Generator[str]:
        '''
        Query-based summarization

        Arguments
        ---
        query: str
            The query
        text: str
            The text to summarize
        cache_prompt: bool
            Cache the summarization input. Defaults to ```False```. This is different from the system prompt caching.
        '''
        pass

    @abstractmethod
    def cache_system_prompt(self):
        '''
        Sends a request with only the system prompt and no text generation. The purpose of this is to cache the system prompt.
        '''
        pass

    @abstractmethod
    def disconnect(self):
        '''
        Disconnect from the LLM
        '''
        pass

    @classmethod
    def prompt(self, query: str, text: str):
        return  self.system_prompt + f"\n### Now summarize the following:\nQuery: \"{query}\"\nFacts:\n{text}\nSummary:\n"

#================================================================================================
    
class LlamaCppSession(LLMSession):

    def __init__(self, model_name: str = "meta-llama-3.1-8b-instruct", api_host="localhost:8080"):
        if api_host is None:
            api_host = "localhost:8080"
        self.model_name = model_name
        self.api_host = api_host

    #-------------------------------------------------------------------------------------------------

    def cache_system_prompt(self):
        resp = requests.post(f"http://{self.api_host}/chat/completions", headers={'Content-Type': 'application/json'}, json={
            'model': "llama",
            'messages': [
                { "role": "user", "content": self.system_prompt }
            ],
            'max_completion_tokens': 0,
            'cache_prompt': True
        })

    #-------------------------------------------------------------------------------------------------

    def disconnect(self):
        self.connection_obj.close()

    #-------------------------------------------------------------------------------------------------

    def summarize(self, query: str, text: str, stop_dict, *, cache_prompt: bool = False) -> Generator[str]:
        stop_dict['conn'] = self
        
        stream = requests.post(f"http://{self.api_host}/chat/completions", stream=True, headers={'Content-Type': 'application/json'}, json={
            'model': "llama",
            'messages': [
                { "role": "user", "content": self.prompt(query, text) }
            ],
            'cache_prompt': cache_prompt,
            'stream': True,
        })

        self.connection_obj = stream
            
        for line in stream.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data_str = decoded[len("data: "):]
                    if data_str == "[DONE]":
                        break

                    #Extract content from SSE message
                    event = json.loads(data_str)
                    content = event['choices'][0]['delta'].get('content', None)
                    if content is not None:
                        yield str(content)
        

#================================================================================================
    
class LMStudioSession(LLMSession):

    def __init__(self, model_name: str = "meta-llama-3.1-8b-instruct", api_host="localhost:1234"):
        if api_host is None:
            api_host = "localhost:1234"
        self.model_name = model_name

        res = re.match(r"(\w+):(\d+)", api_host)

        if res.group(1) == "localhost":
            #Check if I'm using WSL
            #If so, LMStudio is running the windows host, which is the gateway
            if 'microsoft' in platform.uname().release.lower():
                gateway = netifaces.gateways()['default'][netifaces.AF_INET][0]
                self.api_host = f"{gateway}:{res.group(2)}"
            else:
                self.api_host = api_host

        client = lms.Client(api_host=self.api_host)
        self.model = client.llm.model(model_name)

    #-------------------------------------------------------------------------------------------------

    def cache_system_prompt(self):
        chat = lms.Chat()
        chat.add_user_message(self.system_prompt)
        stream = self.model.respond_stream(chat, config=LlmPredictionConfig(max_tokens=1))
        for fragment in stream:
            pass

    #-------------------------------------------------------------------------------------------------

    def disconnect(self):
        self.connection_obj.cancel()
        self.connection_obj.close()

    #-------------------------------------------------------------------------------------------------

    def summarize(self, query: str, text: str, stop_dict, *, cache_prompt: bool = False) -> Generator[str]:
        stop_dict['conn'] = self

        chat = lms.Chat()
        chat.add_user_message(self.prompt(query, text))
        stream = self.model.respond_stream(chat)
        self.connection_obj = stream

        for fragment in stream:
            yield fragment.content
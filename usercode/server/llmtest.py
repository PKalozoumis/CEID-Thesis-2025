from llama_cpp import Llama

llm = Llama(model_path="../../models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf", verbose=False, n_ctx=4096, chat_format="llama-3")

llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."}
    ]
)

output = llm(
      "Q: Hello how are you today? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
)

#output = llm("Q: What is the capital of France? A:", stop=["<|eot_id|>"], max_tokens=)

print(output)
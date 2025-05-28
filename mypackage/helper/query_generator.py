import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


def generate_query(text: str):
    '''
    Generates a query from the given document

    Arguments
    ----------------------------------------
    text: str
        The document
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco').to(device)

    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    outputs = model.generate(
        input_ids=input_ids,
        max_length=64,
        do_sample=True,
        top_k=10,
        num_return_sequences=2
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(res)
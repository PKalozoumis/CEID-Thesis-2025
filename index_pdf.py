from elastic import elasticsearch_client
import base64
from index import create_index
from elasticsearch import NotFoundError

index_name = "pdf-index"
pipeline_name = "pdf-pipeline"

with open("transformer.pdf", "rb") as pdf:
    b64_str = base64.b64encode(pdf.read()).decode("utf-8")

client = elasticsearch_client()
create_index(client, index_name, "pdf_mapping.json")

pipeline = {
        "description": "Pipeline for indexing PDF files",
        "processors": [{
            "attachment":{
                "description": "Index PDF files",
                "field": "data",
                "target_field": "pdf",
                "indexed_chars": -1,
                "remove_binary": True
            }
        }
    ]
}

try:
    client.ingest.delete_pipeline(id=pipeline_name)
except NotFoundError:
    print("Pipeline did not exist")
else:
    print("Pipeline existed")

print(client.ingest.put_pipeline(id=pipeline_name, body=pipeline))

#Add pdf to the index

doc = {
    "data": b64_str
}

res = client.index(index=index_name, body=doc, pipeline=pipeline_name)
print(res)
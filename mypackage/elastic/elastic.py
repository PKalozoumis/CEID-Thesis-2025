import json
from elasticsearch import Elasticsearch, AuthenticationException
import sys
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#================================================================================================================

def elasticsearch_client(credentials_path: str = "credentials.json", cert_path: str = "http_ca.crt") -> Elasticsearch:

    with open(credentials_path, "r") as f:
        credentials = json.load(f)

    client = Elasticsearch(
        "https://localhost:9200",
        ca_certs=cert_path,
        basic_auth=(credentials['elastic_user'], credentials['elastic_password'])
    )\
    .options(ignore_status=400)

    try:
        info = client.info()

    except AuthenticationException:
        print("Wrong password idiot")
        sys.exit()

    return client

#===============================================================================================

if __name__ == "__main__":
    pass
    '''

    from rich.panel import Panel
    from rich.console import Console

    console = Console()
    
    session = Session("arxiv-index")
    docs = [
            Panel(
                ElasticDocument(session, i, filter_path="_source.article_id,_source.summary", text_path="_source.summary").text(),
                title="Text",
                title_align="left",
                border_style="cyan"
            )
            for i in range(1)
        ]

    console.print(docs[0])
    '''
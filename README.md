## Run

### Indexing
```
cd usercode/preprocess
python index.py
```
- Requires:
  - Elasticsearch connection
  - Credentials file: ```credentials.json```
  - Certificate file: ```http_ca.crt```

### Preprocessing
```
cd usercode/preprocess/
python preprocess.py
```
- Requires:
  - Elasticsearch connection (+ files)
  - MongoDB connection

### Application
- First, run server:
```
cd usercode/app/
python server.py
```
- Server requires:
  -  llama.cpp or LMStudio connection
  -  Elasticsearch connection (+ files)
  -  MongoDB connection
- Run client:
```
cd usercode/app/
python console_client.py
```

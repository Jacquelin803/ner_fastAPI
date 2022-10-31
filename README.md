## Named Entity Recognition Web Service and Train Demo  
### Framework  in Project 
+ FastAPI==0.85.1
+ Docker==20.10.18
+ Sqlite(SQLAlchemy==1.3.19)
+ BiLSTM-CRF(torchcrf==1.1.0)
+ Torch==1.7.1

### Code Structure
The codes are structured in the following manner:
+ **ner_train** folder: contains codes for the BiLSTM-CEF train process, cpu/gpu ok
    + **input** folder: small raw data files 
    + **output** folder: processed data which can be used by training, and modelfile(on cpu/gpu) 
    + **model** folder: main train line,contains data process ,net, training, validation
    + how to train: run train_validate.py
+ **ner_app** folder: contains codes for the web service
    + **service** folder: get front request, load modelfile to recognize, return entities to web
    + **db** folder: contains SQLAlchemy models and Pydantic schemas
    + **template** folder: simple front html
+ run.py, Dockerfile, requirements.txt

### How to get web service
#### method 1: python run.py
+ via http://192.168.126.150:8000/docs to check returned data
+ via http://192.168.126.150:8000/ner_app/ front ui, to type sentences and get entities
      (192.168.126.150 = localhost)
#### method 2: docker run
+ get requirements.txt: in root directory terminal type 
```shell script
pipreqs ./ --encoding=utf8 
```   

+ Dockerfile  

+ Build  
```shell script
docker build -t ner_service .
```

+ run
```shell script
docker run -d --name mynerservice -p 80:80 ner_service  
docker ps
```




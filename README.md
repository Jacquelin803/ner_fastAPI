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
+ via http://192.168.126.150:8020/docs to check returned data
+ via http://192.168.126.150:8020/ner_app/ front ui, to type sentences and get entities
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
docker run -it --name mynerservice -p 8020:8020 ner_service  
docker ps
```
+ run main function(run.py) in the container
```shell script
[hadoop@localhost ~]$ sudo docker exec -it  22948981fb4e(CONTAINER ID) /bin/bash
root@22948981fb4e:/code# ls
ner_app  requirements.txt  run.py
root@22948981fb4e:/code# python run.py
/usr/local/lib/python3.7/site-packages/torch/cuda/init.py:52: UserWarning: CUDA initialization: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx (Triggered internally at  /pytorch/c10/cuda/CUDAFunctions.cpp:100.)
return torch._C._cuda_getDeviceCount() > 0
2022-10-27 11:43:42,744 INFO sqlalchemy.engine.base.Engine SELECT CAST('test plain returns' AS VARCHAR(60)) AS anon_1
2022-10-27 11:43:42,744 INFO sqlalchemy.engine.base.Engine ()
2022-10-27 11:43:42,745 INFO sqlalchemy.engine.base.Engine SELECT CAST('test unicode returns' AS VARCHAR(60)) AS anon_1
2022-10-27 11:43:42,745 INFO sqlalchemy.engine.base.Engine ()
2022-10-27 11:43:42,745 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("sentence_entyties")
........
22-10-27 11:43:42,761 INFO sqlalchemy.engine.base.Engine ()
INFO:     Will watch for changes in these directories: ['/code']
INFO:     Uvicorn running on http://0.0.0.0:8020 (Press CTRL+C to quit)
INFO:     Started reloader process [14] using statreload
```
+ web url
http://192.168.126.150:8020/ner_app/ 

(ner_model updating~)


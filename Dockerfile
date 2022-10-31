FROM python:3.7
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./run.py /code/run.py
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
COPY ./ner_app /code/ner_app
EXPOSE 8000
# CMD ["uvicorn","ner_app.run:app","--host","0.0.0.0","--port","8000"]
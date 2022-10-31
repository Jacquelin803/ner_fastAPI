
import uvicorn
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from ner_app import application
from ner_app.db import engine, Base


app = FastAPI()
templates = Jinja2Templates(directory="ner_app/templates")
Base.metadata.create_all(bind=engine)

app.include_router(application, prefix='/ner_app', tags=['命名实体识别首页'])



if __name__ == '__main__':
    uvicorn.run('run:app', host='0.0.0.0', port=8030, reload=True, debug=True, workers=1)
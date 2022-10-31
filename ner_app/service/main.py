
from fastapi import Form

from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from ner_app.db import engine, Base
from ner_app.db.database import SessionLocal
from ner_app.db.models import SentenceEntities
from ner_app.service.ner_predict import NER_Predict

application = APIRouter()

templates = Jinja2Templates(directory="ner_app/templates")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@application.post("/", response_class=HTMLResponse)
async def root(request: Request, data: str = Form(...)):
    print("data============",data)
    result = NER_Predict().predict(data)

    db = SessionLocal()
    db_sentence = SentenceEntities(text=data, entyties=result)
    db.add(db_sentence)
    db.commit()
    print("==========main:",result)
    return templates.TemplateResponse("home.html", {
        "request": request,
        "data": data,
        "result": result,
    })


@application.get("/")
async def main():
    return FileResponse("ner_app/templates/home.html")


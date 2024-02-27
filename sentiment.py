from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline('sentiment-analysis')

@app.get("/")
def root():
    return {'message':'Hello, world!!!'}

@app.post("/predict/")
def predict(item:Item):
    return classifier(item.text)[0]

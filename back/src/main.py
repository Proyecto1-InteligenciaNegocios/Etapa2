from typing import Optional
from fastapi import FastAPI
import pandas as pd
from back.src.DataModel import DataModel
from src.services import classification

app = FastAPI()


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel)
    result = classification.classify_single_text(df)
    return result

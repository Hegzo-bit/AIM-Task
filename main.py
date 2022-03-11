from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import re

app = FastAPI()
class dialect_type(BaseModel):
      text : object 

@app.post('/predict')
async def predict_dialect(txt_1: dialect_type):
          data = txt_1.dict()
          loaded_model = pickle.load(open('SGDClassifier.pkl', 'rb'))
          data_in = [data['text']]
          prediction = loaded_model.predict(data_in)
          prediction1 = prediction.tolist()
          return {
                  'prediction': prediction1,
                  }

                  
from fastapi import FastAPI

import numpy as np
import pandas as pd
import joblib


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/name={name}")
def get_deck(name: str):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    features = []
    for letter in letters:
        features.append(pd.Series([name]).str.count(letter).astype(int)[0])
    regr = joblib.load("model.v0.pickle")
    pred = regr.predict(np.array([features]))[0]
    return {"sexe": int(pred)}

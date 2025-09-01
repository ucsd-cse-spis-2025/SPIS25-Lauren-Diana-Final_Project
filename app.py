from flask import Flask, request
import torch
from SPIS_Stock_Final import model, predict_next_day, scaler 
from SPIS_Stock_Final import tesla_path, tesla_names

app = Flask(__name__)

@app.route('/')
def predict_tesla():
    prediction, confidence = predict_next_day(model, scaler, tesla_path, tesla_names)
    
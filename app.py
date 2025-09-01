from flask import Flask, request, render_template
import torch
from SPIS_Stock_Final import model, predict_next_day, scaler 
from SPIS_Stock_Final import tesla_path, tesla_names, apple_path, apple_names, nvidia_path, nvidia_names, google_path, google_names, meta_path, meta_names, qc_path, qc_names, ms_path, ms_names, amazon_path, amazon_names, samsung_path, samsung_names, netflix_path, netflix_names

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/tesla')
def predict_tesla():
    prediction, confidence = predict_next_day(model, scaler, tesla_path, tesla_names)
    return f"Tesla Prediction: {prediction} (Confidence: {confidence:.2f})"

@app.route('/predict/apple')
def predict_apple():
    prediction, confidence = predict_next_day(model, scaler, apple_path, apple_names)
    return f"Apple Prediction: {prediction} (Confidence: {confidence:.2f})"

@app.route('/predict/nvidia')
def predict_nvidia():
    prediction, confidence = predict_next_day(model, scaler, nvidia_path, nvidia_names)
    return f"Nvidia Prediction: {prediction} (Confidence: {confidence:.2f})"

@app.route('/predict/google')
def predict_google():
    prediction, confidence = predict_next_day(model, scaler, google_path, google_names)
    return f"Google Prediction: {prediction} (Confidence: {confidence:.2f})"

@app.route('/predict/meta')
def predict_meta():
    prediction, confidence = predict_next_day(model, scaler, meta_path, meta_names)
    return f"Meta Prediction: {prediction} (Confidence: {confidence:.2f})"

@app.route('/predict/qualcomm')
def predict_qualcomm():
    prediction, confidence = predict_next_day(model, scaler, qc_path, qc_names)
    return f"Qualcomm Prediction: {prediction} (Confidence: {confidence:.2f})"

@app.route('/predict/microsoft')
def predict_microsoft():
    prediction, confidence = predict_next_day(model, scaler, ms_path, ms_names)
    return f"Microsoft Prediction: {prediction} (Confidence: {confidence:.2f})"

@app.route('/predict/amazon')
def predict_amazon():
    prediction, confidence = predict_next_day(model, scaler, amazon_path, amazon_names)
    return f"Amazon Prediction: {prediction} (Confidence: {confidence:.2f})"

@app.route('/predict/samsung')
def predict_samsung():
    prediction, confidence = predict_next_day(model, scaler, samsung_path, samsung_names)
    return f"Samsung Prediction: {prediction} (Confidence: {confidence:.2f})"

@app.route('/predict/netflix')
def predict_netflix():
    prediction, confidence = predict_next_day(model, scaler, netflix_path, netflix_names)
    return f"Netflix Prediction: {prediction} (Confidence: {confidence:.2f})"
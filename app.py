from flask import Flask, render_template
from SPIS_Stock_Final import model, predict_next_day, scaler 
from SPIS_Stock_Final import tesla_path, tesla_names, apple_path, apple_names, nvidia_path, nvidia_names, google_path, google_names, meta_path, meta_names, qc_path, qc_names, ms_path, ms_names, amazon_path, amazon_names, samsung_path, samsung_names, netflix_path, netflix_names

app = Flask(__name__)

@app.route('/')
def render_home():
    return render_template('index.html')

@app.route('/predict')
def render_main():
    return render_template('main.html')

@app.route('/predict/tesla')
def predict_tesla():
    prediction, confidence = predict_next_day(model, scaler, tesla_path, tesla_names)
    return render_template('tesla.html', prediction=prediction, confidence=round(confidence, 2))

@app.route('/predict/apple')
def predict_apple():
    prediction, confidence = predict_next_day(model, scaler, apple_path, apple_names)
    return render_template('apple.html', prediction=prediction, confidence=round(confidence, 2))

@app.route('/predict/nvidia')
def predict_nvidia():
    prediction, confidence = predict_next_day(model, scaler, nvidia_path, nvidia_names)
    return render_template('nvidia.html', prediction=prediction, confidence=round(confidence, 2))

@app.route('/predict/google')
def predict_google():
    prediction, confidence = predict_next_day(model, scaler, google_path, google_names)
    return render_template('google.html', prediction=prediction, confidence=round(confidence, 2))

@app.route('/predict/meta')
def predict_meta():
    prediction, confidence = predict_next_day(model, scaler, meta_path, meta_names)
    return render_template('meta.html', prediction=prediction, confidence=round(confidence, 2))

@app.route('/predict/qualcomm')
def predict_qualcomm():
    prediction, confidence = predict_next_day(model, scaler, qc_path, qc_names)
    return render_template('qualcomm.html', prediction=prediction, confidence=round(confidence, 2))

@app.route('/predict/microsoft')
def predict_microsoft():
    prediction, confidence = predict_next_day(model, scaler, ms_path, ms_names)
    return render_template('microsoft.html', prediction=prediction, confidence=round(confidence, 2))

@app.route('/predict/amazon')
def predict_amazon():
    prediction, confidence = predict_next_day(model, scaler, amazon_path, amazon_names)
    return render_template('amazon.html', prediction=prediction, confidence=round(confidence, 2))

@app.route('/predict/samsung')
def predict_samsung():
    prediction, confidence = predict_next_day(model, scaler, samsung_path, samsung_names)
    return render_template('samsung.html', prediction=prediction, confidence=round(confidence, 2))

@app.route('/predict/netflix')
def predict_netflix():
    prediction, confidence = predict_next_day(model, scaler, netflix_path, netflix_names)
    return render_template('netflix.html', prediction=prediction, confidence=round(confidence, 2))\


if __name__ == "__main__":
    app.run(host='0.0.0.0')
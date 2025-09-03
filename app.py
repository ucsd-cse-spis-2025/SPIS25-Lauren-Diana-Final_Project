from flask import Flask, render_template, request
from SPIS_Stock_Final import model, predict_from_input, scaler

app = Flask(__name__)

@app.route('/')
def render_home():
    return render_template('index.html')

@app.route('/predict')
def render_main():
    return render_template('main.html')

@app.route('/predict/<stock>', methods=['GET'])
def render_form(stock):
    stock_name = stock.capitalize()
    return render_template('form.html', stock_name=stock_name, stock_route=f"/predict/{stock}/result")

@app.route('/predict/<stock>/result', methods=['POST'])
def handle_prediction(stock):
    stock_name = stock.capitalize()
    stock_ref = stock.lower()

    try:
        open_price = float(request.form['open'])
        high = float(request.form['high'])
        low = float(request.form['low'])
        close = float(request.form['close'])
        volume = float(request.form['volume'])
        date_str = request.form['date']
        assert (open_price > 0 and high > 0 and low > 0 and close > 0 and volume > 0)
    except (KeyError, ValueError) as e:
        return "Invalid input", 400

    prediction, confidence = predict_from_input(stock_ref, model, scaler, open_price, high, low, close, volume, date_str)

    return render_template('result.html',
                           stock_name=stock_name,
                           prediction=prediction,
                           confidence=round(confidence, 2))

                           
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
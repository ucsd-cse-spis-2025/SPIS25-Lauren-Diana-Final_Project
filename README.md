# SPIS25-Lauren-Diana-Final_Project
About SPIS Stocks

Overview
SPIS Stocks is a web app designed to predict whether popular tech stocks will go UP or DOWN the next day using historical price data.

The Model
Our model is a Multi-Layer Perceptron (MLP) built with PyTorch, a popular deep learning framework.
It uses six added features that come from stock data:

open-close: difference between opening and closing prices.

low-high: difference between daily low and high prices.

daily_return: percentage change in closing price compared to previous day.

is_quarter_end: binary flag indicating if the date is at the end of a fiscal quarter.

stock_id: numeric identifier encoding the stock symbol.

The MLP consists of three hidden layers with 128, 64, and 32 neurons respectively, using ReLU activations and dropout layers to help prevent overfitting.
The model outputs a probability indicating whether a stock will go up (1) or down (0) the following day.

Data
We train our model on a combined dataset collected from Kaggle, featuring historical daily data from ten major stocks: Tesla, Apple, Nvidia, Google, Meta, Qualcomm, Microsoft, Amazon, Samsung, and Netflix.
Each dataset contains typical stock market columns like Date, Open, High, Low, Close, and Volume.

Note that some datasets are a bit outdated and don’t reflect the most recent market activity, but they provide a solid foundation for training the model.

Predictions
Using the previous day's data, the model predicts if a stock’s closing price will move UP or DOWN that day.
The output also includes a confidence score representing how sure the model is of its prediction.

While our model performs slightly better than random chance, with an overall accuracy around 52%, it’s important to remember that stock markets are highly complex and influenced by many unpredictable factors.

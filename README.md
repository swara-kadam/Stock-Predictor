# 📊 Stock Predictor Web App

This is a simple Flask-based web app that predicts whether selected stocks are likely to go 📈 up or 📉 down based on their past 6 months of price data.

## 🚀 Features
- Multi-stock prediction (select up to 5)
- Uses Yahoo Finance data via `yfinance`
- Machine learning model (Logistic Regression)
- 6-month price charts generated with Matplotlib
- Simple web UI using Flask & Jinja2

## 🧠 How it works
- Fetches historical closing prices for each stock
- Trains a logistic regression model to predict next-day direction
- Renders chart + prediction + simple explanation

## 🛠 Technologies Used
- Python
- Flask
- scikit-learn
- yfinance
- pandas & matplotlib

## 📦 Installation

```bash
git clone git@github.com:your-username/stock-predictor-app.git
cd stock-predictor-app
pip install -r requirements.txt
python app.py

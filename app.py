from flask import Flask, request, render_template
import yfinance as yf
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') #sets the backend, some of them use a GUI environment. when running flask on a server (headless), we need a non interactive backend. agg renders as png

import matplotlib.pyplot as plt
import io
import base64   


TICKERS_DF = pd.read_csv("tickers.csv")
TICKERS = TICKERS_DF.to_dict(orient='records')


app = Flask(__name__)

def generate_chart(df, symbol):
    plt.figure(figsize=(6,3))
    df['Close'].plot(title=f'{symbol} - Last 6 Months')
    plt.xlabel("Date")
    plt.ylabel("Price")

    buf = io.BytesIO() #temp storage area in memory - write and read without touching disk
    plt.savefig(buf, format='png') #saves to buffer as a png
    plt.close() #close current plot
    buf.seek(0) #moves r/w pointer to start of buffer to read it
    return base64.b64encode(buf.read()).decode('utf-8') #reads the binary data from buffer, encodes as a base64 text string, then decode as utf8
#encoding avoids opening gui window at all, where no graphical interface is available. integrate seamlessly into web based workflow

def train_model(stock_symbol):
    df = yf.download(stock_symbol, period="6mo")
    if df.empty:
        raise ValueError("No data returned")

    df['Target'] = df['Close'].shift(-1) > df['Close']
    df['Target'] = df['Target'].astype(int)
    df['Close_shifted'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    X = df[['Close_shifted']]
    y = df['Target']

    if y.nunique() < 2: # should have 0 and 1 
        raise ValueError("Not enough class variety")

    model = LogisticRegression()
    model.fit(X, y)

    latest_price = df['Close'].iloc[-1].item()  # converts pandas series format to regular numpy number
    pred = model.predict(np.array([[latest_price]]))[0]
    prediction = "📈 Going Up" if pred == 1 else "📉 Going Down"
    average_price = df['Close'].mean().item()  #force float
    explanation = f"Latest price (${latest_price:.2f}) vs avg (${average_price:.2f}) → likely {prediction.split()[1]}."

    chart = generate_chart(df, stock_symbol)
    return {
        'symbol': stock_symbol,
        'prediction': prediction,
        'reason': explanation,
        'chart': chart
    }


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        symbols = request.form.getlist("stocks")
        for symbol in symbols[:5]:
            try:
                result = train_model(symbol)
                results.append(result)
            except Exception as e:
                print(f"Error with {symbol}: {e}")
    return render_template("index.html", results=results, tickers=TICKERS)


if __name__ == "__main__":
    app.run(debug=True)

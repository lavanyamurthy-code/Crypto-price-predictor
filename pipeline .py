

import os
import json
import requests
import pandas as pd
import numpy as np
import torch
import ta

from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


import time
import yfinance as yf
import streamlit as st

from sklearn.ensemble import RandomForestRegressor



def fetch_price(
    symbol="BTC-USD",
    years=1,
    save_path="data/prices.csv"
):

    print(f"📥 Fetching {years} years of daily data from Yahoo Finance...")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{years}y", interval="1d")
    except Exception as e:
        print("❌ Yahoo Finance error:", e)
        return None

    if df.empty:
        print("❌ No data fetched from Yahoo.")
        return None

    df.reset_index(inplace=True)

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    df.columns = ["date", "open", "high", "low", "close", "volume"]

    os.makedirs("data", exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"✅ Saved {len(df)} rows to {save_path}")

    return df




NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
if not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY not found! Add it in HF Secrets.")


def fetch_news():

    print("Fetching filtered Bitcoin news...")

    url = "https://newsapi.org/v2/everything"

    params = {
        # Stronger search query
        "q": '"Bitcoin price" OR "BTC price" OR "BTC USD" OR "Bitcoin market"',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params).json()
    articles = response.get("articles", [])

    filtered_data = []

    strong_keywords = [
        "bitcoin",
        "btc",
        "btc/usd",
        "crypto market",
        "bitcoin price"
    ]

    for a in articles:

        title = (a.get("title") or "").lower()
        description = (a.get("description") or "").lower()

        # 🔎 Strict filtering: Bitcoin must appear in TITLE
        if not any(word in title for word in ["bitcoin", "btc"]):
            continue

        # 🔎 Relevance scoring
        combined_text = title + " " + description
        score = sum(combined_text.count(word) for word in strong_keywords)

        if score < 2:
            continue

        filtered_data.append({
            "date": a["publishedAt"][:10],
            "text": a["title"] + ". " + str(a.get("description"))
        })

    df = pd.DataFrame(filtered_data)

    df.to_csv("data/news.csv", index=False)

    return df



def fetch_live_price(coin="bitcoin"):
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": coin,
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_last_updated_at": "true"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        price = data[coin]["usd"]
        change_24h = data[coin]["usd_24h_change"]
        last_updated = datetime.fromtimestamp(
            data[coin]["last_updated_at"]
        )

        return {
            "price": price,
            "change_24h": change_24h,
            "last_updated": last_updated
        }

    except Exception as e:
        raise RuntimeError(f"Live price fetch failed: {e}")



@st.cache_resource
def load_sentiment_model():
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    return model, tokenizer

sentiment_model, sentiment_tokenizer = load_sentiment_model()
sentiment_model.eval()

device = torch.device("cpu")
sentiment_model.to(device)

def analyze_sentiment():
    print("Fetching sentiment...")
    news = pd.read_csv("data/news.csv")  # full path

    sentiments = []
    for text in news["text"].astype(str):
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k:v.to(device) for k,v in inputs.items()}

        with torch.no_grad():
            outputs = sentiment_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        score = (probs[0, 2] - probs[0, 0]).item()  # positive - negative
        sentiments.append(score)

    news["sentiment"] = sentiments
    news.to_csv("data/sentiment.csv", index=False)
    print("✅ sentiment.csv created with", len(sentiments), "rows")
    return news


def detect_trend(price_csv="data/prices.csv"):
    df = pd.read_csv(price_csv)
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df = df.dropna(subset=["rsi"])
    return df.iloc[-1]

def compute_avg_sentiment(sentiment_csv="data/sentiment.csv"):
    df = pd.read_csv(sentiment_csv)
    return df["sentiment"].mean()
    
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found! Add it in HF Secrets.")


class CryptoPrediction(BaseModel):
    direction: str = Field(description="UP, DOWN, or NEUTRAL")
    expected_change_percent: float = Field(description="Expected % price change")
    confidence: int = Field(description="Confidence from 0 to 100")
    reason: str = Field(description="Short explanation")

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a professional crypto market analyst. "
     "Respond ONLY with valid JSON. "
     "The 'reason' field MUST contain at least 2-3 detailed lines explaining the prediction clearly."
    ),
    ("human", """
Bitcoin Market Data:

- Current LIVE Price: {live_price}
- ML Predicted Tomorrow Price: {ml_price}
- News Sentiment Score: {sentiment}
- RSI: {rsi}

Recent News Headlines:
{news}

Instructions:
1. Determine direction relative to LIVE price (UP, DOWN, or NEUTRAL).
2. Estimate expected percentage change from LIVE price.
3. Provide confidence (0–100).
4. Give a clear 2-3 line explanation using price trend, RSI, and news context.
""")
])

'''prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a crypto market analyst. Respond ONLY with valid JSON."),
    ("human", """
Bitcoin indicators:
- Today's Close Price: {price}
- News Sentiment Score: {sentiment}
- RSI: {rsi}
Predict tomorrow's price movement.
""")
])
'''

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


def llm_predict_tomorrow(live_price, ml_pred, sentiment, rsi, news):

    print("Fetching LLM prediction...")

    chain = prompt | llm.with_structured_output(CryptoPrediction)

    try:
        result = chain.invoke({
            "live_price": live_price,
            "ml_price": ml_pred,
            "sentiment": sentiment,
            "rsi": rsi,
            "news": news
        })

    except ValidationError as e:
        print(f"Validation error caught: {e}")

        return CryptoPrediction(
            direction='NEUTRAL',
            expected_change_percent=0.0,
            confidence=0,
            reason='LLM parsing error or unexpected output.'
        ).model_dump()

    return result.model_dump()
    



def rf_ml_predict(price_csv="data/prices.csv"):

    df = pd.read_csv(price_csv)

    # ----- Feature Engineering -----
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["ema"] = ta.trend.EMAIndicator(df["close"], window=14).ema_indicator()

    df["close_lag1"] = df["close"].shift(1)
    df["close_lag2"] = df["close"].shift(2)
    df["close_lag3"] = df["close"].shift(3)

    df["return_1"] = df["close"].pct_change(1)
    df["return_2"] = df["close"].pct_change(2)

    df["rolling_mean_5"] = df["close"].rolling(5).mean()
    df["rolling_std_5"] = df["close"].rolling(5).std()

    df["target"] = df["close"].shift(-1)

    df.dropna(inplace=True)

    features = [
        "open","high","low","close","volume",
        "rsi","macd","ema",
        "close_lag1","close_lag2","close_lag3",
        "return_1","return_2",
        "rolling_mean_5","rolling_std_5"
    ]

    # ==============================
    # WALK-FORWARD TRAINING
    # ==============================
    model = None

    for i in range(60, len(df)-1):

        train = df.iloc[:i]

        X_train = train[features]
        y_train = train["target"]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

    # ==============================
    # FINAL PREDICTION (Tomorrow)
    # ==============================
    X_last = df[features].iloc[[-1]]
    tomorrow_price = model.predict(X_last)[0]

    return round(tomorrow_price, 2)




def predict_btc_7days():

    # ==============================
    # FETCH DATA
    # ==============================

    df = yf.download("BTC-USD", interval="1d", period="3y")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)

    # ==============================
    # FEATURES
    # ==============================

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()

    df["macd"] = df["ema20"] - df["ema50"]

    df["lag1"] = df["Close"].shift(1)
    df["lag2"] = df["Close"].shift(2)
    df["lag3"] = df["Close"].shift(3)

    df["return_1d"] = df["Close"].pct_change()
    df["target"] = df["return_1d"].shift(-1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    features = [
        "Open","High","Low","Volume",
        "rsi","macd",
        "lag1","lag2","lag3"
    ]

    # ==============================
    # TRAIN MODEL
    # ==============================

    split = int(len(df) * 0.8)

    X_train = df[features].iloc[:split]
    y_train = df["target"].iloc[:split]

    model = RandomForestRegressor(
        n_estimators=800,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ==============================
    # FUTURE PREDICTION
    # ==============================

    future_df = df.copy()

    future_predictions = []

    df['high_close_ratio'] = df['High'] / df['Close']
    df['low_close_ratio'] = df['Low'] / df['Close']
    df['open_close_ratio'] = df['Open'] / df['Close']

    avg_high_close_ratio = df['high_close_ratio'].mean()
    avg_low_close_ratio = df['low_close_ratio'].mean()
    avg_open_close_ratio = df['open_close_ratio'].mean()
    avg_volume = df['Volume'].mean()

    for i in range(7):

        last_row = future_df.iloc[-1]

        X_future = pd.DataFrame([last_row[features]], columns=features)

        pred_return = model.predict(X_future)[0]

        new_price = last_row["Close"] * (1 + pred_return)

        new_row = last_row.copy()

        new_row["Close"] = new_price

        ratio_noise = np.random.normal(1,0.005)
        volume_noise = np.random.normal(1,0.05)

        new_row["Open"] = last_row["Close"] * avg_open_close_ratio * ratio_noise
        new_row["High"] = new_price * avg_high_close_ratio * ratio_noise
        new_row["Low"] = new_price * avg_low_close_ratio * ratio_noise
        new_row["Volume"] = avg_volume * volume_noise

        new_row["lag1"] = new_price
        new_row["lag2"] = last_row["lag1"]
        new_row["lag3"] = last_row["lag2"]

        future_df = pd.concat(
            [future_df, pd.DataFrame([new_row])]
        )

        future_df["rsi"] = ta.momentum.RSIIndicator(
            future_df["Close"], window=14
        ).rsi()

        ema20 = future_df["Close"].ewm(span=20).mean()
        ema50 = future_df["Close"].ewm(span=50).mean()

        future_df["macd"] = ema20 - ema50

        future_predictions.append(new_price)

    return future_predictions

    



def rf_hourly_24_predict(hours=24):

    # =====================
    # Download Data
    # =====================
    df = yf.download(
        "BTC-USD",
        interval="1h",
        period="120d"
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]

    df.dropna(inplace=True)

    # =====================
    # Feature Engineering
    # =====================

    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_3"] = df["close"].shift(3)

    df["hour"] = df.index.hour
    df["day"] = df.index.dayofweek

    # target = next hour close
    df["target"] = df["close"].shift(-1)

    df.dropna(inplace=True)

    # =====================
    # Features
    # =====================

    features = [
        "open",
        "high",
        "low",
        "volume",
        "lag_1",
        "lag_2",
        "lag_3",
        "hour",
        "day"
    ]

    X = df[features]
    y = df["target"]

    # =====================
    # Train Model on FULL DATA
    # =====================

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # =====================
    # 24 Hour Prediction
    # =====================

    predictions = []

    last_row = df.iloc[-1:].copy()

    for step in range(hours):

        X_latest = last_row[features]

        next_price = model.predict(X_latest)[0]

        predictions.append(round(next_price, 2))

        # update lag values
        last_row["lag_3"] = last_row["lag_2"]
        last_row["lag_2"] = last_row["lag_1"]
        last_row["lag_1"] = next_price

        last_row["close"] = next_price

        last_row["hour"] = (last_row["hour"] + 1) % 24
        last_row["day"] = (last_row["day"] + (last_row["hour"] == 0)) % 7

    return predictions




def predict_btc_7days():

    # ==============================
    # FETCH DATA
    # ==============================

    df = yf.download("BTC-USD", interval="1d", period="3y")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)

    # ==============================
    # FEATURES
    # ==============================

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()

    df["macd"] = df["ema20"] - df["ema50"]

    df["lag1"] = df["Close"].shift(1)
    df["lag2"] = df["Close"].shift(2)
    df["lag3"] = df["Close"].shift(3)

    df["return_1d"] = df["Close"].pct_change()
    df["target"] = df["return_1d"].shift(-1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    features = [
        "Open","High","Low","Volume",
        "rsi","macd",
        "lag1","lag2","lag3"
    ]

    # ==============================
    # TRAIN MODEL
    # ==============================

    split = int(len(df) * 0.8)

    X_train = df[features].iloc[:split]
    y_train = df["target"].iloc[:split]

    model = RandomForestRegressor(
        n_estimators=800,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ==============================
    # FUTURE PREDICTION
    # ==============================

    future_df = df.copy()

    future_predictions = []

    df['high_close_ratio'] = df['High'] / df['Close']
    df['low_close_ratio'] = df['Low'] / df['Close']
    df['open_close_ratio'] = df['Open'] / df['Close']

    avg_high_close_ratio = df['high_close_ratio'].mean()
    avg_low_close_ratio = df['low_close_ratio'].mean()
    avg_open_close_ratio = df['open_close_ratio'].mean()
    avg_volume = df['Volume'].mean()

    for i in range(7):

        last_row = future_df.iloc[-1]

        X_future = pd.DataFrame([last_row[features]], columns=features)

        pred_return = model.predict(X_future)[0]

        new_price = last_row["Close"] * (1 + pred_return)

        new_row = last_row.copy()

        new_row["Close"] = new_price

        ratio_noise = np.random.normal(1,0.005)
        volume_noise = np.random.normal(1,0.05)

        new_row["Open"] = last_row["Close"] * avg_open_close_ratio * ratio_noise
        new_row["High"] = new_price * avg_high_close_ratio * ratio_noise
        new_row["Low"] = new_price * avg_low_close_ratio * ratio_noise
        new_row["Volume"] = avg_volume * volume_noise

        new_row["lag1"] = new_price
        new_row["lag2"] = last_row["lag1"]
        new_row["lag3"] = last_row["lag2"]

        future_df = pd.concat(
            [future_df, pd.DataFrame([new_row])]
        )

        future_df["rsi"] = ta.momentum.RSIIndicator(
            future_df["Close"], window=14
        ).rsi()

        ema20 = future_df["Close"].ewm(span=20).mean()
        ema50 = future_df["Close"].ewm(span=50).mean()

        future_df["macd"] = ema20 - ema50

        future_predictions.append(new_price)

    return future_predictions



def run_pipeline():
    os.makedirs("data", exist_ok=True)

    # ---------- 1. Fetch price ----------
    # ---------- 1. Fetch price ----------
    df = fetch_price(
        symbol="BTC-USD",
        years=3
    )

    if df is None:
        raise RuntimeError("Price fetch failed. prices.csv not created.")

    try:
        live_data = fetch_live_price()
        live_price = live_data["price"]
    except Exception as e:
        print("⚠️ Live price fetch failed:", e)
        live_price = df["close"].iloc[-1]



    # ---------- 2. Fetch news ----------
    fetch_news()

    # ---------- 3. Sentiment ----------
    analyze_sentiment()
    avg_sentiment = compute_avg_sentiment()

    # ---------- 4. Trend ----------
    today = detect_trend()

    # ---------- 5. ML Prediction ----------
    ml_pred = rf_ml_predict()

    hourly_predict = rf_hourly_24_predict()

    btc_7days =predict_btc_7days()
    
    news_df = fetch_news()

    # 2️⃣ Handle empty case
    if news_df.empty:
        top_news = "No major Bitcoin news today."
    else:
        top_news = "\n".join(news_df["text"].head(3).tolist())
      

    # ---------- 6. LLM Prediction ----------
    '''llm_result = llm_predict_tomorrow(
        today["close"],
        avg_sentiment,
        today["rsi"]
    )'''


    llm_result = llm_predict_tomorrow(
        live_price=live_price,
        ml_pred=ml_pred,
        sentiment=avg_sentiment,
        rsi=today["rsi"],
        news=top_news
    )
    llm_pred_price = today["close"] * (1 + llm_result["expected_change_percent"] / 100)

    # ---------- 7. Final Weighted Hybrid ----------
    final_price = (0.75 * ml_pred) + (0.25 * llm_pred_price)

    # ---------- 8. Save Output ----------
    output = {
        "current_price": round(live_price, 2),
        "ml_predicted_price": round(ml_pred, 2),
        "llm_predicted_price": round(llm_pred_price, 2),
        "final_hybrid_price": round(final_price, 2),
        "direction": llm_result["direction"],
        "confidence": llm_result["confidence"] / 100,
        "reason": llm_result["reason"],
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hourly_predict": hourly_predict,
        "seven_day_pred" : btc_7days
    }

    with open("data/prediction.json", "w") as f:
        json.dump(output, f, indent=4)



if __name__ == "__main__":
    run_pipeline()




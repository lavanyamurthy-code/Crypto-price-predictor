import os
import json
import requests
import pandas as pd
import streamlit as st
import pipeline

import yfinance as yf
import plotly.graph_objects as go
import ta

from datetime import datetime, timedelta

import streamlit as st
from database import create_db, add_user, login_user

from chatbot import generate_chat_response


create_db()



if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    st.markdown('<p class="title-text"></p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.title("Bitcoin Price Predictor")
        
        st.markdown('<div class="login-box">', unsafe_allow_html=True)

        menu = ["Login","Register"]

        choice = st.radio(
            " ",
            menu,
            horizontal=True   # ⭐ makes radio buttons side by side
        )

        if choice == "Register":

            st.subheader("Create Account")

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Register"):
                try:
                    add_user(username, password)
                    st.success("Account created successfully!")
                except:
                    st.error("User already exists")

        elif choice == "Login":

            st.subheader("Login")

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):

                result = login_user(username, password)

                if result:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        st.markdown('</div>', unsafe_allow_html=True)


        
if st.session_state.logged_in:

    st.success("Welcome to Bitcoin Predictor Dashboard 🚀")



    # Your existing pipeline code
    # pipeline.run_pipeline()


    @st.cache_data(ttl=3600)
    def fetch_btc_data(years):
        end = datetime.now()

        if years == "MAX":
            start = datetime(2010, 1, 1)
        else:
            start = end - timedelta(days=365 * years)

        df = yf.download(
            "BTC-USD",
            start=start,
            end=end,
            interval="1d",
            progress=False
        )

        if df.empty:
            raise ValueError("No BTC data returned")

        # 🔥 FIX 1: Remove MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)

        # 🔥 FIX 2: Force 1D arrays
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        return df



    import plotly.graph_objects as go
    import json

    # =============================
    # FUNCTIONS
    # =============================

    @st.cache_data(ttl=3600)
    def load_hourly_history():
        hist = yf.download("BTC-USD", interval="1h", period="7d")
        hist = hist.reset_index()

        if "Close" in hist.columns:
            hist["Price"] = hist["Close"]
        elif ("Close", "BTC-USD") in hist.columns:
            hist["Price"] = hist[("Close", "BTC-USD")]

        if "Datetime" not in hist.columns:
            hist.rename(columns={"Date": "Datetime"}, inplace=True)

        return hist[["Datetime", "Price"]]


    @st.cache_data(ttl=3600)
    def load_daily_history():
        hist = yf.download("BTC-USD", interval="1d", period="30d")
        hist = hist.reset_index()

        if "Close" in hist.columns:
            hist["Price"] = hist["Close"]
        elif ("Close", "BTC-USD") in hist.columns:
            hist["Price"] = hist[("Close", "BTC-USD")]

        return hist[["Date", "Price"]]


    def plot_chart(hist, df_pred, x_col):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist[x_col],
            y=hist["Price"],
            mode="lines",
            name="Historical Price",
            line=dict(color="blue")
        ))

        fig.add_trace(go.Scatter(
            x=df_pred[x_col],
            y=df_pred["Price"],
            mode="lines+markers",
            name="Predicted Price",
            line=dict(color="orange")
        ))

        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title="BTC Price (USD)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)


    # ---------------- PAGE CONFIG ----------------
    st.set_page_config(
        page_title="Crypto Predictor",
        page_icon="📊",
        layout="wide"
    )
    
    

    with st.sidebar.container():
         # ---------------- SIDEBAR NAV ----------------
        page = st.sidebar.radio(
            "📌 Navigate",
            ["Overview", "🔮 Prediction","📈 Price Chart", "📰 Bitcoin News", "🤖 AI Chatbot"]
        )
        st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        # Logout button at bottom
        if st.sidebar.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # ======================================================
    # 🔮 PAGE 1 — PREDICTION DASHBOARD
    # ======================================================
    if page == "Overview":

        st.subheader("Bitcoin Market Prediction")

        if st.button("🔄 Update Prediction"):
            with st.spinner("Running prediction pipeline..."):
                pipeline.run_pipeline()
                st.success("Prediction updated successfully!")

        if not os.path.exists("data/prediction.json"):
            st.error("Prediction data not found. Run the pipeline.")
            st.stop()

        with open("data/prediction.json") as f:
            data = json.load(f)
        st.session_state.btc_chat_context = data

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "💰 Current Price",
            f"${data['current_price']}"
        )

        col2.metric(
            "🔮 Predicted Price (Tomorrow)",
            f"${data['final_hybrid_price']}",
            delta=f"{(data['final_hybrid_price'] - data['current_price']) / data['current_price'] * 100:.2f}%"
        )

        col3.metric(
            "📈 Direction",
            data["direction"]
        )

        st.divider()

        confidence_percent = int(data["confidence"] * 100)

        st.subheader("📊 Model Confidence")
        st.progress(data["confidence"])
        st.write(f"**Confidence Level:** {confidence_percent}%")

        if confidence_percent >= 70:
            st.success("High confidence prediction 🔥")
        elif confidence_percent >= 40:
            st.warning("Moderate confidence ⚠️")
        else:
            st.error("Low confidence ❗")

        st.divider()

        st.subheader("🧠 Reasoning")
        st.write(data["reason"])

        st.caption(f"⏱️ Last Updated: {data['last_updated']}")

    # ======================================================
    # 📈 PAGE 2 — BITCOIN PRICE CHART
    # ======================================================
    elif page == "📈 Price Chart":

        import plotly.graph_objects as go
        import pandas as pd
        from datetime import datetime

        st.subheader("📊 Bitcoin TradingView-Style Chart")

        # ============================
        # RANGE MODE
        # ============================
        range_mode = st.radio(
            "Select Range Mode",
            ["Preset Range", "Custom Date Range"],
            horizontal=True
        )

        range_map = {
            "1 Year": 1,
            "3 Years": 3,
            "5 Years": 5,
            "10 Years": 10,
            "MAX": "MAX"
        }

        # ============================
        # INPUTS
        # ============================
        if range_mode == "Preset Range":
            selected = st.selectbox("Select Time Range", list(range_map.keys()))
            years = range_map[selected]
            start_date = None
            end_date = None

        else:
            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=pd.to_datetime("2024-01-01")
                )

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=pd.to_datetime("2025-01-01")
                )

            if start_date >= end_date:
                st.error("Start date must be before end date")
                st.stop()

            years = "MAX"  # fetch full data once

        # ============================
        # FETCH DATA
        # ============================
        try:
            df = fetch_btc_data(years)
            df["Date"] = pd.to_datetime(df["Date"])

            # ============================
            # FILTER DATA (CUSTOM RANGE)
            # ============================
            if range_mode == "Custom Date Range":
                df = df[
                    (df["Date"] >= pd.to_datetime(start_date)) &
                    (df["Date"] <= pd.to_datetime(end_date))
                ]

            if df.empty:
                st.warning("No data available for selected date range.")
                st.stop()

            # ============================
            # CANDLESTICK CHART
            # ============================
            price_fig = go.Figure()

            price_fig.add_trace(
                go.Candlestick(
                    x=df["Date"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="BTC Price"
                )
            )

            price_fig.update_layout(
                title="Bitcoin Price (USD)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                height=520
            )

            st.plotly_chart(price_fig, use_container_width=True)

            # ============================
            # RSI CHART
            # ============================
            rsi_fig = go.Figure()

            rsi_fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["rsi"],
                    name="RSI (14)",
                    line=dict(color="orange")
                )
            )

            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")

            rsi_fig.update_layout(
                title="Relative Strength Index (RSI)",
                yaxis_title="RSI",
                xaxis_title="Date",
                template="plotly_dark",
                height=260
            )

            st.plotly_chart(rsi_fig, use_container_width=True)

            # ============================
            # CAPTION
            # ============================
            if range_mode == "Custom Date Range":
                st.caption(
                    f"Showing BTC-USD from "
                    f"{start_date.strftime('%d %b %Y')} to "
                    f"{end_date.strftime('%d %b %Y')}"
                )
            else:
                st.caption(f"Showing BTC-USD for {selected}")

        except Exception as e:
            st.error("Failed to load TradingView-style chart")
            st.code(str(e))



    # ======================================================
    # 📰 PAGE 3 — BITCOIN NEWS
    # ======================================================
    elif page == "📰 Bitcoin News":

        st.subheader("📰 Latest Bitcoin News")

        NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

        if not NEWS_API_KEY:
            st.error("NEWS_API_KEY not found in environment.")
            st.stop()

        url = "https://newsapi.org/v2/everything"

        params = {
            "q": '"Bitcoin price" OR "BTC price" OR "BTC USD" OR "Bitcoin market"',
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": NEWS_API_KEY
        }

        with st.spinner("Fetching latest Bitcoin news..."):
            r = requests.get(url, params=params).json()
            articles = r.get("articles", [])

        if not articles:
            st.warning("No Bitcoin news articles found.")
            st.stop()

        # Strong filtering
        strong_keywords = ["bitcoin", "btc", "btc/usd", "crypto"]

        for article in articles:

            title = (article.get("title") or "").lower()
            description = (article.get("description") or "").lower()

            # 🔎 Must mention Bitcoin in TITLE
            if not any(word in title for word in ["bitcoin", "btc"]):
                continue

            # 🔎 Relevance score check
            combined_text = title + " " + description
            score = sum(combined_text.count(word) for word in strong_keywords)

            if score < 2:
                continue

            st.markdown(f"### 📰 {article['title']}")
            st.write(article.get("description", ""))
            st.markdown(f"[Read full article]({article['url']})")
            published = article["publishedAt"].replace("Z", "")
            st.caption(f"Source: {article['source']['name']} | {published}")
            st.divider()

    
    # =============================
    # PREDICTION PAGE
    # =============================

    elif page == "🔮 Prediction":

        st.subheader("📈 Bitcoin Prediction Dashboard")

        mode = st.radio(
            "Select Prediction Mode",
            ["24 Hour Prediction", "7 Day Prediction"],
            horizontal=True
        )

        with open("data/prediction.json") as f:
            data = json.load(f)

        # =============================
        # 24 HOUR PREDICTION
        # =============================
        if mode == "24 Hour Prediction":

            hour_choice = st.slider("Select Hour Ahead", 1, 24, 1)

            if st.button("Show Hourly Prediction", key="hour_btn"):

                with st.spinner("Fetching hourly BTC data..."):
                    hist = load_hourly_history()

                predictions = data["hourly_predict"]

                last_time = hist["Datetime"].iloc[-1]
                last_price = hist["Price"].iloc[-1]

                future_times = pd.date_range(
                    start=last_time + pd.Timedelta(hours=1),
                    periods=hour_choice,
                    freq="H"
                )

                df_pred = pd.DataFrame({
                    "Datetime": future_times,
                    "Price": predictions[:hour_choice]
                })

                df_pred = pd.concat([
                    pd.DataFrame({"Datetime": [last_time], "Price": [last_price]}),
                    df_pred
                ], ignore_index=True)

                pred_price = predictions[hour_choice - 1]
                change = pred_price - last_price

                col1, col2, col3 = st.columns(3)
                
                col1.metric("Predicted Price", f"${pred_price:,.2f}")
                

                plot_chart(hist, df_pred, "Datetime")

                st.subheader("Hourly Forecast Table")
                st.dataframe(df_pred.set_index("Datetime"))

        # =============================
        # 7 DAY PREDICTION
        # =============================
        elif mode == "7 Day Prediction":

            if st.button("Show 7 Day Prediction", key="day_btn"):

                with st.spinner("Fetching daily BTC data..."):
                    hist = load_daily_history()

                predictions = data["seven_day_pred"]

                last_time = hist["Date"].iloc[-1]
                last_price = hist["Price"].iloc[-1]

                future_times = pd.date_range(
                    start=last_time + pd.Timedelta(days=1),
                    periods=7,
                    freq="D"
                )

                df_pred = pd.DataFrame({
                    "Date": future_times,
                    "Price": predictions
                })

                df_pred = pd.concat([
                    pd.DataFrame({"Date": [last_time], "Price": [last_price]}),
                    df_pred
                ], ignore_index=True)

                pred_price = predictions[-1]
                change = pred_price - last_price

                col1, col2, col3 = st.columns(3)
                col1.metric("7th Day Prediction", f"${pred_price:,.2f}")
    

                plot_chart(hist, df_pred, "Date")

                st.subheader("7 Day Forecast Table")
                st.dataframe(df_pred.set_index("Date"))  
            

    # ======================================================
    # 🤖 PAGE 5 — AI CHATBOT
    # ======================================================
    elif page == "🤖 AI Chatbot":

        from chatbot import generate_chat_response

        df_daily = yf.download("BTC-USD", period="40d", interval="1d")

        yesterday_close = df_daily["Close"].iloc[-2]
        yesterday_high = df_daily["High"].iloc[-2]
        month_high = df_daily["High"].max()
        month_low = df_daily["Low"].min()

        st.session_state.btc_chat_context.update({
            "y_close": float(yesterday_close),
            "y_high": float(yesterday_high),
            "month_high": float(month_high),
            "month_low": float(month_low)
        })

        st.subheader("🤖 Bitcoin AI Market Analyst")

        # ---------- Load Prediction Context ----------
        if not os.path.exists("data/prediction.json"):
            st.warning("⚠️ Run prediction from Overview first")
            st.stop()

        with open("data/prediction.json") as f:
            ctx = json.load(f)

        # ---------- Clear Chat Button ----------
        col1, col2 = st.columns([6,1])

        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        st.write("💬 Ask anything about BTC")

        # ---------- Chat Memory ----------
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, msg in st.session_state.chat_history:
            st.chat_message(role).write(msg)

        # ---------- Chat Input ----------
        user_query = st.chat_input("Ask about BTC prediction")

        if user_query:

            response = generate_chat_response(user_query, ctx)

            st.session_state.chat_history.append(("user", user_query))
            st.session_state.chat_history.append(("assistant", response))

            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(response)

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os


# ---------- LOAD GEMINI MODEL ----------
@st.cache_resource
def load_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    return llm


# ---------- CHAT RESPONSE FUNCTION ----------
def generate_chat_response(user_query, ctx):

    llm = load_llm()
    prompt = f"""
You are an advanced Crypto Market Analyst AI.

Your job is to explain Bitcoin market prediction clearly using trading logic.

================ MARKET DATA ================

Current Price: {ctx['current_price']}

ML Predicted Price: {ctx['ml_predicted_price']}
LLM Predicted Price: {ctx['llm_predicted_price']}
Final Hybrid Prediction: {ctx['final_hybrid_price']}

Prediction Direction: {ctx['direction']}
Model Confidence: {ctx['confidence']}

Hourly Forecast: {ctx['hourly_predict']}
7 Day Forecast: {ctx['seven_day_pred']}

Yesterday Close: {ctx.get('y_close')}
Yesterday High: {ctx.get('y_high')}

Monthly High: {ctx.get('month_high')}
Monthly Low: {ctx.get('month_low')}



Reason For Prediction:
{ctx['reason']}

================ RULES ================

- Explain in simple trading language
- Always justify direction using price logic
- Compare prediction with current price
- Mention risk when confidence is low
- If market is sideways → say no clear trade
- Do NOT guarantee profits
- Provide educational guidance only

User Question: {user_query}

Answer:
"""

    response = llm.invoke(prompt).content

    return response
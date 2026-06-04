import streamlit as st
from database import login_user



st.markdown("""
<style>

.title-text{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#2c7be5;
    margin-bottom:20px;
}

.login-box{
    background-color:#f7f9fc;
    padding:35px;
    border-radius:15px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}

.stButton>button{
    width:100%;
    border-radius:8px;
    background-color:#2c7be5;
    color:white;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#1a5edb;
}

</style>
""", unsafe_allow_html=True)


def login():

    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        result = login_user(username,password)

        if result:
            st.session_state["logged_in"] = True
            st.success("Login successful")
        else:
            st.error("Invalid credentials")
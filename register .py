import streamlit as st
from database import add_user

def register():

    st.title("Register")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Register"):

        add_user(username,password)

        st.success("Account created successfully!")
import streamlit as st
import pandas as pd

st.title("Help")

st.markdown(
    """
    In construction.
    """
)

data = pd.read_csv("example/results.csv")

st.write(data)

with open("example/results.csv", "rb") as file:
    button = st.download_button(
        label="Download",
        data=file,
        file_name="results.csv"
    )
import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import requests

load_dotenv()
API_SERVER_URL = os.getenv("API_SERVER_URL")


def similar_n(cut, color, clarity, carat, n):
    payload = {
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "carat": carat,
        "n": n
    }
    response = requests.get(
        f"http://{API_SERVER_URL}/n-similar",
        params=payload
    )
    diamonds = response.json()
    return pd.DataFrame(diamonds)


def predict(cut, color, clarity, carat, depth, table, x, y, z):
    payload = {
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "carat": carat,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
    }
    response = requests.get(
        f"http://{API_SERVER_URL}/prediction",
        params=payload
    )
    return response.json()


st.write("# Diamonds")
st.write("---")

st.write("## Predict price")
st.write(
    (
        "Given diamond data, it will predict the price "
        "using the best model loaded in the server."
    )
)

cols1 = st.columns(5, vertical_alignment="bottom")
cols2 = st.columns(5, vertical_alignment="bottom")

pred_cut = cols1[0].selectbox(
    "Cut", ["Fair", "Good", "Very Good", "Ideal", "Premium"], key="pred_cut"
)
pred_color = cols1[1].selectbox(
    "Color", ["D", "E", "F", "G", "H", "I", "J"], key="pred_color"
)
pred_clarity = cols1[2].selectbox(
    "Clarity",
    ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
    key="pred_clarity",
)
pred_carat = cols1[3].number_input("Carat", step=0.01, key="pred_carat")
pred_depth = cols1[4].number_input("Depth", step=0.1, key="pred_depth")
pred_table = cols2[0].number_input("Table", step=1, key="pred_table")
pred_x = cols2[1].number_input("X", step=0.01, key="pred_x")
pred_y = cols2[2].number_input("Y", step=0.01, key="pred_y")
pred_z = cols2[3].number_input("Z", step=0.01, key="pred_z")
pred_btn = cols2[4].button("Predict")

if pred_btn:
    pred_price = predict(
        pred_cut,
        pred_color,
        pred_clarity,
        pred_carat,
        pred_depth,
        pred_table,
        pred_x,
        pred_y,
        pred_z,

    )
    st.write(f"The predicted price is :blue-background[{pred_price:.2f}]")

st.write("---")
st.write("## Find *n* similar")
st.write(
    (
        "Given *cut*, *color*, and *clarity*, and *carat* of a diamond "
        "display the *n* most similar diamonds acording to weight."
    )
)

cols = st.columns(6, vertical_alignment="bottom")

cut = cols[0].selectbox(
    "Cut",
    ["Fair", "Good", "Very Good", "Ideal", "Premium"]
)
color = cols[1].selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = cols[2].selectbox(
    "Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]
)
carat = cols[3].number_input("Carat", step=0.01)
n = cols[4].number_input("*n*", min_value=0, step=1, value=5)
find_btn = cols[5].button("Find")

if find_btn:
    st.dataframe(similar_n(cut, color, clarity, carat, n))

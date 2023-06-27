import streamlit as st
import pandas as pd

st.title("Uber Map")
DATE_COLUMN = "date/time"


# @st.cache_data
def load_data():
    df = pd.read_csv("./data/uber.csv")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


# 데이터 로딩하기
data_load_state = st.text("Loading data...")
df = load_data()
data_load_state.text("Loading data...done!")

st.subheader("Map of all pickups")
hour_to_filter = st.slider("hour", 0, 23, 15)
filtered_df = df[df[DATE_COLUMN].dt.hour == hour_to_filter]
st.map(filtered_df)

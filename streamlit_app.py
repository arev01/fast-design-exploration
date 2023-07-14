import streamlit as st
import pandas as pd
import numpy as np

st.title('Uber pickups in NYC')

@st.cache_data
def load_data(filename):
    data = pd.read_csv(filename)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

FILE_TYPES = ["csv", "py", "png", "jpg"]

file_in = st.file_uploader("Upload file", type=FILE_TYPES)
show_file = st.empty()
if not file_in:
    show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))
    
data_load_state = st.text('Loading data...')
data = load_data(file_in)
data_load_state.text("Done! (using st.cache_data)")

nb_params = len(data.columns)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# Some number in the range 0-23
for name in data.columns:
    min = data[name].min()
    max = data[name].max()
    
    filter = st.slider(name, min, max, min)
#filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

#st.subheader('Map of all pickups at %s:00' % hour_to_filter)
#st.map(filtered_data)

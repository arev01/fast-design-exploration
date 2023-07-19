import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.title("Uber pickups in NYC")

@st.cache_data
def load_data(file_name):
    data = pd.read_csv(file_name)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

FILE_TYPES = ["csv"]

uploaded_files = st.sidebar.file_uploader("Upload files", type=FILE_TYPES, accept_multiple_files=True)
show_file = st.empty()
if len(uploaded_files) == 2:

    option = st.sidebar.selectbox("Select input file", [f.name for f in uploaded_files])

    df_in = load_data(option)

    df_out = load_data([f.name for f in uploaded_files if f.name != option][0])

    df = pd.concat([df_in, df_out.reindex(df_in.index)], axis=1)

    if st.checkbox("Show raw data"):
        st.write(df_in)
        st.write(df_out)

    tabs = st.tabs(df_out.columns.values.tolist())

    columns = st.sidebar.multiselect("Select columns", df_in.columns)

    if len(columns) > 0:
        for k, val in enumerate(df_out.columns):

            # creating feature variables
            X = df_in[columns]

            with tabs[k]:
                figures = []

                for name in X.columns:
                    fig = px.scatter(df, x=name, y=val, trendline="ols", trendline_color_override="red")
                    figures.append(fig)
                    
                if st.checkbox("Show scatter plots"):
                    columns = st.columns(len(figures))
                    for i in range(len(columns)):
                        columns[i].plotly_chart(figures[i])
                
                # define dataset
                X_old = X.to_numpy()
                y_old = df_out[val]
                # define the model
                model = RandomForestRegressor()
                # fit the model
                model.fit(X_old, y_old)
                # get importance
                importance = model.feature_importances_
                if st.checkbox("Show histogram"):
                    # plot feature importance
                    hist_values = np.histogram(importance)[0]
                    st.bar_chart(hist_values)
                    # summarize feature importance
                    for i,v in enumerate(importance):
                        st.text("Feature: %0d, Score: %.5f" % (i,v))

                #st.subheader('Number of pickups by hour')

                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                from sklearn import preprocessing
                
                # creating train and test sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X_old, y_old, test_size=0.3, random_state=101)
                
                # creating a regression model
                model = LinearRegression()
                
                # fitting the model
                model.fit(X_train, y_train)
                
                # making predictions
                predictions = model.predict(X_test)
                
                # model evaluation
                #st.text(mean_squared_error(y_test, predictions))
                #st.text(mean_absolute_error(y_test, predictions))

                if st.checkbox("Display XY plot"):
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(x=y_test, y=predictions, mode="markers"))

                    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines"))

                    st.plotly_chart(fig)

                if st.checkbox("Make prediction"):
                    # define one new data instance
                    X_new = []
                    for name in X.columns:
                        filter = st.slider(label=name,
                                            min_value=float(df_in[name].min()),
                                            step=0.01,
                                            max_value=float(df_in[name].max()),
                                            value=float(df_in[name].min()),
                                            format="%f"
                                            )
                    
                        X_new.append(filter)
                    
                    # make a prediction
                    y_new = model.predict([X_new])

                    st.slider(label=val,
                            min_value=float(df_out[val].min()),
                            step=1.0,
                            max_value=float(df_out[val].max()),
                            value=float(y_new[0]),
                            format="%f"
                            )

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

st.write("# Fast Design Exploration")

st.markdown(
    """
    In construction.
    """
)

@st.cache_data
def load_data(file_name):
    data=pd.read_csv(file_name)
    lowercase=lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    return data

in_file=st.file_uploader("Upload file", type="csv")

if in_file:
    df=load_data(in_file)

    if st.checkbox("Show raw data"):
        st.write(df)

        st.markdown("---")

    params=st.multiselect("Select parameters", df.columns.tolist())

    if len(params) > 0:
        df_updated=df.copy()

    parameter_values={}
    exp_explore=st.sidebar.expander(label="Parameter Space Exploration")
    for name in list(df_updated[params].keys()):
        slider_ph = exp_explore.empty()
        options=sorted(list(df_updated[name].unique()))
        parameter_values[name] = slider_ph.select_slider(label=name, options=options,value=[min(options),max(options)])#,key=name)

        df_updated=df_updated[((df_updated[name]>=parameter_values[name][0]) & (df_updated[name]<=parameter_values[name][1]))]

    if len(params) > 0:
        if st.checkbox("Show scatter plot"):
            paramx1=st.selectbox("Select parameter for x-axis", options=df.columns.values, key="paramx1")
            paramy1=st.selectbox("Select parameter for y-axis", options=df.columns.values, key="paramy1")

            trace11=go.Scatter(x=df[paramx1], y=df[paramy1], name="full dataset", mode="markers", marker=dict(color="blue",size=10))
            fig1=go.Figure(data=trace11)
            
            if not df_updated.empty:
                trace21=go.Scatter(x=df_updated[paramx1], y=df_updated[paramy1], name="selected", mode="markers", marker=dict(color="red",size=15))
                fig1.add_trace(trace21)

            st.plotly_chart(fig1)#, use_container_width=True)

            st.markdown("---")
            
        if st.checkbox("Show histogram"):
            target=st.selectbox("Select target", df.drop(params, axis=1).columns.values, key="target")
            
            # feature_importance
            X=np.array(df_updated[params])
            y=np.array(df_updated[target])
            #model=LinearRegression()
            model=DecisionTreeRegressor()
            model.fit(X,y)
            #importances=model.coef_
            importances=model.feature_importances_

            df_bar=pd.DataFrame(data=np.column_stack([params,importances.T.tolist()]), columns=["feature","importance"])
            #st.write(df_bar)
            
            fig2=px.bar(df_bar, x="feature", y="importance")
            st.plotly_chart(fig2)#, use_container_width=True)

            st.markdown("---")

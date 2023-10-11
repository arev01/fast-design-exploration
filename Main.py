import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from st_pages import Page, show_pages, add_page_title

from Read_Write_CSV import *

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

# Specify what pages should be shown in the sidebar, and what their titles 
# and icons should be
show_pages(
    [
        Page("Main.py", "Home", "🏠"),
        Page("pages/1_Help.py", "Help", ":question:"),
        Page("pages/2_Config.py", "Config", ":gear:"),
    ]
)

st.write("# Fast Design Exploration")

st.markdown(
    """
    Online tool for screening and analysis of simulation results.  
    
    ### Getting Started  
    **👈 Need help to get started?** Go to the 'Help' section from the sidebar and learn the easy steps needed to analyze your simulation results.  
    **👈 Hungry for more?** Go to the 'Config' section from the sidebar and unlock advanced functionalities.
    """
)

if "separator" not in st.session_state:
    st.session_state["separator"] = ","
    
if "decimal" not in st.session_state:
    st.session_state["decimal"] = "."

in_file=st.file_uploader("Upload file", type="csv")

if in_file:
    dict_vars=read_variables_csv(in_file)
    df=pd.DataFrame(dict_vars)
    df_updated=df.copy()

    key_lst=st.sidebar.multiselect("Choose filters", options=df.columns.values)

    if len(key_lst) > 0:
        params={key: None for key in key_lst}
        mode=st.sidebar.selectbox("Select mode", options=["By value","By range","By min/max"])

        if mode == "By value":
            for key in key_lst:
                p=st.sidebar.select_slider(label=key, options=sorted(list(df[key].unique())))
                df_updated=df_updated[(df[key]==p)]

        elif mode == "By range":
            for key in key_lst:
                p=st.sidebar.select_slider(label=key, options=sorted(list(df[key].unique())), value=[df[key].min(), df[key].max()])
                df_updated=df_updated[((df[key]>=p[0]) & (df[key]<=p[1]))]

        elif mode == "By min/max":
            for key in key_lst:
                params[key]=not st.sidebar.toggle(key)
                #st.sidebar.write(params[key])

            key=key_lst[0]
            if params[key]:
                df_sorted=df.sort_values(by=[key], ascending=True).reset_index(drop=True)
            else:
                df_sorted=df.sort_values(by=[key], ascending=False).reset_index(drop=True)

            df_updated=df_sorted.iloc[:1]
                
            for key in key_lst:
                for i in range(1,len(df_sorted)):
                    bool_lst=[]
                    if params[key]:
                        bool_lst.append((df_sorted[key].iloc[i]<=df_updated[key].loc[len(df_updated.index)-1]))
                    else:
                        bool_lst.append((df_sorted[key].iloc[i]>=df_updated[key].loc[len(df_updated.index)-1]))

                    if all(bool_lst):
                            df_updated.loc[len(df_updated.index)]=df_sorted.iloc[i].tolist()

    if st.checkbox("Show raw data"):
        st.dataframe(df)

        st.markdown("---")
    
    st.write("### Scatter plot")
    
    p_x1=st.selectbox("Select parameter for x-axis", options=df.columns.values, key="p_x1")
    p_y1=st.selectbox("Select parameter for y-axis", options=df.columns.values, key="p_y1")

    trace11=go.Scatter(x=df[p_x1], y=df[p_y1], name="unfiltered", mode="markers", marker=dict(color="blue",size=10))
    fig1=go.Figure(data=trace11)
    fig1.update_xaxes(title=p_x1)
    fig1.update_yaxes(title=p_y1)
    
    if not df_updated.empty:
        trace21=go.Scatter(x=df_updated[p_x1], y=df_updated[p_y1], name="selected", mode="markers", marker=dict(color="red",size=15))
        fig1.add_trace(trace21)

    st.plotly_chart(fig1)#, use_container_width=True)

    st.markdown("---")

    st.write("### Line chart")

    p_y2=st.selectbox("Choose parameter for y-axis", options=df.columns.values, key="p_y2")

    trace12=go.Scatter(x=None, y=df_updated[p_y2], name="selected", mode="lines", line=dict(color="red",width=10))
    fig2=go.Figure(data=trace12)
    fig2.update_xaxes(title="Design ID")
    fig2.update_yaxes(title=p_y2)

    st.plotly_chart(fig2)#, use_container_width=True)

    st.markdown("---")
        
    st.write("### Feature importance")

    if not df_updated.empty:
        objective=st.selectbox("Select objective function", df.drop(key_lst, axis=1).columns.values, key="target")
    
        if len(key_lst) > 0:
            # feature_importance
            X=np.array(df_updated[key_lst])
            y=np.array(df_updated[objective])
            #model=LinearRegression()
            model=DecisionTreeRegressor()
            model.fit(X,y)
            #importances=model.coef_
            importances=model.feature_importances_

            df_bar=pd.DataFrame(data=np.column_stack([key_lst,importances.T.tolist()]), columns=["feature","importance"])
            #st.write(df_bar)
            
            trace13=px.bar(df_bar, x="feature", y="importance")
            fig3=go.Figure(data=trace13)
            fig3.update_traces(marker_color="red")
            
            st.plotly_chart(fig3)#, use_container_width=True)

        else:
            st.error("Wrong input (no filter selected)")

        st.markdown("---")

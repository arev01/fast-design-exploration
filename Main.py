from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from Read_Write_CSV import *
from Regularization import *
from Bisect import *

st.title("👋 Home")

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

if "settings" not in st.session_state:
    st.session_state["settings"] = "Basic"

in_file=st.file_uploader("Upload file/folder", type=["csv", "png"], accept_multiple_files=True)

if in_file:
    for f in in_file:
        if f.name.endswith(".csv"):
            csv_file=f

    if csv_file:
        dict_vars=read_variables_csv(csv_file)
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
            st.write(df)

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

        st.write("### Clustering")
        
        # Select features for clustering
        features = st.multiselect("Choose columns to use for clustering", options=df.columns.values)
        X = df[features]
        
        # Scale the data (optional)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply the K-Means algorithm
        k = st.number_input("Specify the number of clusters", min_value=1, max_value=10)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init is set to 10 for current sklearn versions
        kmeans.fit(X_scaled)
        
        # Get the cluster labels and add them to the original DataFrame
        df['cluster_label'] = kmeans.labels_
        
        st.write("\nDataFrame with Clusters:")
        st.write(df)

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

        objective=st.selectbox("Select objective function", df.drop(key_lst, axis=1).columns.values, key="target")

        if key_lst:
            # feature_importance
            X=np.array(df[key_lst])
            y=np.array(df[objective])
            #model=LinearRegression()
            model=DecisionTreeRegressor()
            model.fit(X,y)
            #importances=model.coef_
            importances=model.feature_importances_

            df_bar=pd.DataFrame(data=np.column_stack([key_lst,importances.T.tolist()]), columns=["feature","importance"])
            #st.write(df_bar)
            
            trace13=px.bar(df_bar, x="feature", y="importance")
            fig3=go.Figure(data=trace13)
            fig3.update_traces(marker_color="blue")
            
            st.plotly_chart(fig3)#, use_container_width=True)

        st.markdown("---")

        # st.write("### 2D Plots")

        # png_lst=[f for f in in_file if f.name.endswith(".png")]
        # png_lst_sorted=sorted(png_lst, key=lambda x: x.name, reverse=False)

        # n_col=len(png_lst) // df.shape[0]
        # png_lst_sorted=[png_lst_sorted[x:x+n_col] for x in range(0, len(png_lst), n_col)]

        # i_range=range(n_col)
        # pairs=[(x,y) for x in i_range for y in i_range if y != x]

        # type_lst=[]
        # for p in pairs:
        #     answer=""
        #     string1, string2=png_lst_sorted[0][p[0]].name, png_lst_sorted[0][p[1]].name
        #     len1, len2=len(string1), len(string2)
        #     for i in range(len1):
        #         match=""
        #         for j in range(len2):
        #             if (i+j < len1 and string1[i+j] == string2[j]):
        #                 match+=string2[j]
        #             else:
        #                 if (len(match) > len(answer)): answer = match
        #                 match=""

        #     type_lst.append(string1.replace(answer, "")[:-4])
        
        # plot_type=st.selectbox("Select 2D plot type", type_lst, key="target3")

        # col1, col2, col3=st.columns([0.33, 0.33, 0.33])

        # images=Image.open([f for f in in_file if f.name.endswith(plot_type+".png")])
        # #images=[f for f in in_file if f.name.endswith(plot_type+".png")]
        # st.image(images, width=400)

        # st.markdown("---")

        #if st.session_state["settings"] == "Advanced":
        st.write("### Regression")

        objective=st.selectbox("Select objective function", df.drop(key_lst, axis=1).columns.values, key="target2")

        regression=st.selectbox("Select regression method", options=["Polynomial", "Kriging", "Radial-Basis", "Neural-Network", "Decision-Tree"], disabled=True)

        if regression == "Polynomial":
            col1, col2, col3=st.columns([0.33, 0.33, 0.33])

            with col1:
                degree=col1.radio(label="Choose degree", options=[1, 2, 3], horizontal=True, label_visibility="collapsed")
            
            with col2:
                standardize=col2.toggle("Standardize")

            with col3:
                regularize=col3.toggle("Regularize")

            #X is the independent variable
            X=df[key_lst]
            #y is the dependent data
            y=df[objective]

            if standardize == True:
                scaler=StandardScaler().fit(X)
                X=scaler.transform(X)

            #generate a model of polynomial features
            poly=PolynomialFeatures(degree)

            #transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
            X_=poly.fit_transform(X)

            index_lst=list(range(0, X_.shape[1]))

            if regularize == True:
                cross_validator=CustomCrossValidation(
                    loss_function=mean_square_error, 
                    X=X_, Y=y, ModelClass=CustomLinearRegression
                )
                cross_validator.cross_validate()
                X_=cross_validator.X_star
                for idx in cross_validator.X_star_index:
                    index_lst.remove(idx)

            clf=CustomLinearRegression(
                loss_function=mean_square_error,
                X=X_, Y=y
            )
            data=clf.summary()

            if st.checkbox("Show feature summary"):
                features=["x%d" % i for i in index_lst]
                data.index=features
                st.table(data)

                st.markdown("---")

            #st.write(np.round(clf.score(), 3))

            predictions=clf.predict(X_)
            residuals=(y-predictions)/y

            trace14=go.Scatter(x=predictions, y=y, name="unfiltered", mode="markers", marker=dict(color="blue",size=10))
            fig4=go.Figure(data=trace14)
            fig4.add_trace(go.Scatter(x=predictions, y=predictions, name="1:1 line", mode="lines", marker_color="lightgreen"))
            fig4.update_xaxes(title="predictions")
            fig4.update_yaxes(title="targets")
            #fig4.update_layout(showlegend=False)
            
            st.plotly_chart(fig4)#, use_container_width=True)

            # targets=[0]*X_.shape[0]

            # trace15=go.Scatter(x=predictions, y=residuals, name="unfiltered", mode="markers", marker=dict(color="blue",size=10))
            # fig5=go.Figure(data=trace15)
            # fig5.add_trace(go.Scatter(x=predictions, y=targets, name="1:1 line", mode="lines", marker_color="lightgreen"))
            # fig5.update_xaxes(title="predictions")
            # fig5.update_yaxes(title="residuals", tickformat=".0%")
            # #fig5.update_layout(showlegend=False)
            
            # st.plotly_chart(fig5)#, use_container_width=True)

            # df_hist=pd.DataFrame(data=residuals.tolist(), columns=["residuals"])
            # #st.write(df_hist)
            
            # trace16=px.histogram(df_hist, y="residuals", color_discrete_sequence=["blue"])
            # fig6=go.Figure(data=trace16)
            # fig6.update_layout(yaxis_title=None)#, bargap=0.2)
            
            # st.plotly_chart(fig6)#, use_container_width=True)
        
        st.markdown("---")

        st.write("### Prediction")

        steps=st.text_input("Choose number of steps", 5)
        
        try:
            steps=int(steps)
        except:
            st.error("- Wrong input (cannot be converted to integer)")
        
        # Create a mesh grid on which we will run our model
        key_range=[]
        for key in key_lst:
            key_min, key_max=X[key].min(), X[key].max()
            key_range.append(np.linspace(key_min, key_max, steps))#, mesh_size)
        
        mesh_grid=np.meshgrid(*key_range)
        flattened=[mesh.ravel() for mesh in mesh_grid]

        # Run model
        pred=model.predict(np.c_[flattened].T)
        #pred=pred.reshape(mesh_grid.shape)

        df_interp=pd.DataFrame(np.c_[flattened].T, columns=key_lst)
        df_interp[objective]=pred

        if st.checkbox("Show interpolated data"):
            st.write(df_interp)

        out_file=write_csv(df_interp)
        button=st.download_button("Download", out_file, "interp.csv", "text/csv")

        # # Generate the plot
        # trace17=px.scatter_3d(df, x=key_lst[0], y=key_lst[1], z=objective)
        # fig7=go.Figure(data=trace17)
        # fig7.update_traces(marker=dict(size=5))
        # fig7.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name="pred_surface"))

        # st.plotly_chart(fig7)#, use_container_width=True)

        st.markdown("---")


import streamlit as st
import pandas as pd
import plotly.express as px

# load the diabetes CSV file into a dataframe
df = pd.read_csv("diabetes.csv")
# Create a sample dataframe
# data = pd.read_csv('heart-diseasee.csv', dtype=str)
# df=data['Heart Disease']=="True"
#         # Plot!
# df1=data[df]
# st.header('Age Distribution by Heart Disease Status')
# import plotly.express as px
# bins = st.slider('Number of bins:', min_value=10, max_value=50, value=20)
# st.dataframe(df1)
# fig = px.histogram(data[['Age','Heart Disease']], x='Age', color='Heart Disease', nbins=bins)
# st.plotly_chart(fig, use_container_width=True)
# st.subheader("Chest pain type count for heart disease")
# fig1 = px.bar(df1[['Chest pain type']], x='Chest pain type')
# st.plotly_chart(fig1,use_container_width=True)

try:
    columns = st.multiselect("Select columns to graph", df.columns)
    
    # create a scatter plot using plotly
    fig = px.scatter(df, x=columns[0], y=columns[1], color="Outcome")

    # display the chart in streamlit
    st.plotly_chart(fig)
except:
    st.subheader("Insert two attributes  ⬆️")

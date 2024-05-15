import streamlit as st
import pandas as pd
import plotly.express as px

st.title(":signal_strength: :blue[ Let's take a look at the statistics of last 5 years]")
st.write("---")

df = pd.read_csv("data_csv.csv")   

ASD_traits_data=df["ASD_traits"].unique().tolist()
select_date=st.selectbox("ASD traits ?",ASD_traits_data)
df_up=df[df["ASD_traits"].isin(ASD_traits_data)]

sub_opt=df_up["Sex"].unique().tolist()
select_sub=st.multiselect("Gender",sub_opt)
df_up_sub=df_up[df_up["Sex"].isin(select_sub)]
st.write("---")
# First pair of plots
col1, col2 = st.columns(2)
with col1:
    st.subheader("Jaundice statistics")
    with st.expander("See the plot"):     
        fig = px.bar(df_up_sub, x="Sex", color="Jaundice")
        fig.update_layout(height=500, width=200)
        st.write(fig)

with col2:
    st.subheader("Childhood Autism Rating Scale statistics")
    with st.expander("See the plot"):        
        fig = px.bar(df_up_sub, x="Sex", color="Childhood Autism Rating Scale")
        fig.update_layout(height=500, width=200)
        st.write(fig)

st.write("---")

# Second pair of plots
col3, col4 = st.columns(2)
with col3:
    st.subheader("Family member with ASD statistics")
    with st.expander("See the plot"):     
        fig = px.bar(df_up_sub, x="Sex", color="Family_mem_with_ASD")
        fig.update_layout(height=500, width=200)
        st.write(fig)

with col4:
    st.subheader("Social Responsiveness Scale statistics")
    with st.expander("See the plot"):        
        fig = px.bar(df_up_sub, x="Sex", color="Social_Responsiveness_Scale")
        fig.update_layout(height=500, width=200)
        st.write(fig)

st.write("---")

# Third pair of plots
col5, col6 = st.columns(2)
with col5:
    st.subheader("Learning Disorder statistics")
    with st.expander("See the plot"):     
        fig = px.bar(df_up_sub, x="Sex", color="Learning disorder")
        fig.update_layout(height=500, width=200)
        st.write(fig)

with col6:
    st.subheader("Global Developmental Delay/Intellectual Disability statistics")
    with st.expander("See the plot"):        
        fig = px.bar(df_up_sub, x="Sex", color="Global developmental delay/intellectual disability")
        fig.update_layout(height=500, width=200)
        st.write(fig)
# Fourth pair of plots
col7, col8 = st.columns(2)
with col7:
    st.subheader("Depression statistics")
    with st.expander("See the plot"):     
        fig = px.bar(df_up_sub, x="Sex", color="Depression")
        fig.update_layout(height=500, width=200)
        st.write(fig)

with col8:
    st.subheader("Anxiety Disorder statistics")
    with st.expander("See the plot"):        
        fig = px.bar(df_up_sub, x="Sex", color="Anxiety_disorder")
        fig.update_layout(height=500, width=200)
        st.write(fig)
col9, col10 = st.columns(2)
with col9:
    st.subheader("Speech Delay/Language Disorder statistics")
    with st.expander("See the plot"):     
        fig = px.bar(df_up_sub, x="Sex", color="Speech Delay/Language Disorder")
        fig.update_layout(height=500, width=200)
        st.write(fig)

with col10:
    st.subheader("Genetic Disorders statistics")
    with st.expander("See the plot"):        
        fig = px.bar(df_up_sub, x="Sex", color="Genetic_Disorders")
        fig.update_layout(height=500, width=200)
        st.write(fig)

st.subheader("Dataset")
st.write(df)
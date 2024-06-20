import streamlit as st
import textdescriptives as td
import pandas as pd
from openai import OpenAI
from sentence_transformers import CrossEncoder, util, SentenceTransformer
import numpy as np

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; margin-bottom: 20px'>AI Capabilities for KM</h1>", unsafe_allow_html=True)

text = st.text_area("Enter some text here:", height=150, label_visibility="collapsed", placeholder="Enter your text here 📝")

if st.button("Submit"):
    df_descriptives = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["descriptive_stats", "readability","dependency_distance","coherence"])
    df_descriptives_brief = df_descriptives[['gunning_fog', 'flesch_kincaid_grade', 'sentence_length_mean']]
    st.write(" ")
    with st.expander("Readability Metrics"):
        st.write('Original Text Metrics')
        st.dataframe(df_descriptives_brief)

        #rewrite text in simplified form
        prompt = f"""
        Your task is to rewrite the provided text so that it can easily be understood by a high school student. Make sure you include all the same information and keep the same formatting. Do not start your response with 'Text:' or 'Answer:'.

        Text: {text}
        Answer:
        """
        gpt = None
        completion = gpt.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "Your task is to rewrite the provided text so that it can easily be understood by a high school student. Make sure you include all the same information and keep the same formatting. Do not start your response with 'Text:' or 'Answer:' "},
        {"role": "user", "content": prompt}
        ]
        )
        output = completion.choices[0].message.content

        st.write("")
        st.write(f'**Simplified Text:** {output}')

        st.write(" ")
        st.write('Simplified Text Metrics')
        text = output
        df_descriptives_simple = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["descriptive_stats", "readability","dependency_distance","coherence"])
        df_descriptives_simple_brief = df_descriptives_simple[['gunning_fog', 'flesch_kincaid_grade', 'sentence_length_mean']]
        st.dataframe(df_descriptives_simple_brief)

    st.write(" ")

    with st.expander("Similar/Contradictory Texts"):
        st.write('hello')
        
    st.write(" ")

    with st.expander("Cluster Details"):
        st.write('hello')

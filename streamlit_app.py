import streamlit as st
import textdescriptives as td
import pandas as pd
from openai import OpenAI
from sentence_transformers import CrossEncoder, util, SentenceTransformer
import numpy as np
import weaviate
import json

auth_config = weaviate.AuthApiKey(api_key="MutVi6yIYXH5xsoUlxhDH0O4GwO1aBCe1Jz0")

client = weaviate.Client(
  url="https://digest-data-english-i0rn0tsj.weaviate.network",
  auth_client_secret=auth_config
)
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; margin-bottom: 20px'>AI Capabilities for KM</h1>", unsafe_allow_html=True)

text = st.text_area("Enter some text here:", height=150, label_visibility="collapsed", placeholder="Enter your text here 📝")

with st.sidebar:
  api_token = st.text_input("Enter your OpenAI key:", type='password')
    
gpt = OpenAI(api_key=api_token)

def top_results(text):
  model_name = 'sentence-transformers/all-MiniLM-L6-v2'
  vect_model = SentenceTransformer(model_name)
  reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
  query_embedding = vect_model.encode(text)
  response = (
  client.query
  .get("Digest2", ["content", "section_title", "doc_id", "section_chapter", "section_text"])
  .with_hybrid(query=text, vector=query_embedding)
  .with_additional(["score"])
  .with_limit(20)
  .do()
  )
  
  results = []
  for item in response['data']['Get']['Digest2']:
    result = {
        'doc_id': item['doc_id'],
        'section_title': item['section_title'],
        'section_chapter': item['section_chapter'],
        'score': item['_additional']['score'],
        'content': item['content'],
        'section_text': item['section_text']
    }
    results.append(result)

  query_doc_pairs = [[text, res["content"]] for res in response["data"]["Get"]["Digest2"]]

  scores = reranker_model.predict(query_doc_pairs)

  top_n = 10 ### Cap number of documents that are sent to LLM for RAG
  scores_cp = scores.tolist()
  documents = [pair[1] for pair in query_doc_pairs]
  content = ""
  content_display = []

  for _ in range(top_n):
    index = scores_cp.index(max(scores_cp))
    content += documents[index]
    content_display.append(documents[index])

    del documents[index]
    del scores_cp[index]

  content_set = set(content_display)
  doc_display = [docs for docs in results if docs['content'].strip() in content_set]
  df_similar = pd.DataFrame.from_dict(doc_display, orient='columns')
  df_similar = df_similar[['doc_id', 'section_chapter', 'section_title', 'score', 'content']]

  return df_similar, results

def label_conflicts(have_conflicts):
    return have_conflicts


def check_conflicts(doc_pair):
    # Step 1: send the conversation and available functions to the model
    content = """ Is there certain conflict in the information, facts, or assumptions delivered between these two documents?
    Document 1:
    {}

    Document 2:
    {}
    """.format(doc_pair[0], doc_pair[1])
    hint = "The different scenarios, terminologies, and phrasings are NOT conflicts. Be sensitive towards numbers. You must call the function with parameter that indicates if there is conflict."
    messages = [{"role": "user", "content": content}, {"role": "system", "content": hint}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "label_conflicts",
                "description": "To tell user if there are conflicts",
                "parameters": {
                  "type": "object",
                  "properties": {
                      "have_conflicts": {
                          "type": "boolean",
                          "description": "Boolean value, True if there is conflict between two documents, False if there is NO conflict",
                      },
                  },
                  "required": ["have_conflicts"]
              },
            }
        }
    ]
    response = gpt.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        tools=tools,
        seed=123,
        temperature=0,
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "label_conflicts":label_conflicts
        }  # only one function in this example, but you can have multiple
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments) # No Arguments
            function_response = function_to_call(
                have_conflicts=function_args.get("have_conflicts")
            )
            # Make Model Reasoning Transparent:
            if function_response:
              feed_back_prompt = """ You conclude there is certain conflict in the information, facts, or assumptions delivered between these two documents, please explain the reason
              Document 1:
              {}

              Document 2:
              {}
              """.format(doc_pair[0], doc_pair[1])
              hint = "The different scenarios, terminologies, and phrasings are not considered conflicts. Be sensitive towards numbers."
              completion = gpt.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                  {"role": "user", "content": feed_back_prompt},
                  {"role": "system", "content": hint}
                ],
                seed=123,
                temperature=0,
              )
              print("\n")
              print(completion)
              return completion.choices[0].message.content
            else:
               return None
    
if st.button("Submit"):
    with st.expander("Readability Metrics"):
        st.write('Original Text Metrics')
      
        df_descriptives = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["descriptive_stats", "readability","dependency_distance","coherence"])
        df_descriptives_brief = df_descriptives[['gunning_fog', 'flesch_kincaid_grade', 'sentence_length_mean']]
        #st.write(" ")
        st.dataframe(df_descriptives_brief)

        #rewrite text in simplified form
        prompt = f"""
        Your task is to rewrite the provided text so that it can easily be understood by a high school student. Make sure you include all the same information and keep the same formatting. Do not start your response with 'Text:' or 'Answer:'.

        Text: {text}
        Answer:
        """
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
        df_descriptives_simple = td.extract_metrics(text=output, spacy_model="en_core_web_lg", metrics=["descriptive_stats", "readability","dependency_distance","coherence"])
        df_descriptives_simple_brief = df_descriptives_simple[['gunning_fog', 'flesch_kincaid_grade', 'sentence_length_mean']]
        st.dataframe(df_descriptives_simple_brief)

    st.write(" ")

    with st.expander("Similar/Contradictory Texts"):
        st.write('**Similar Texts**')

        #find similar texts
        df_similar, results = top_results(text)
        df_results = pd.DataFrame(results)

        #create document pairs
        res = []
        idx = []
        for i in (range(len(df_results))):
            doc_pair = [text, results[i]['content']]
            idx_pair = ['original_text', results[i]['doc_id']]
            res.append(doc_pair)
            idx.append(idx_pair)
        
        #check for conflicts
        full_docs = []
        for i in range(len(res)):
            pair = res[i]
            conflict = check_conflicts(pair)
            if conflict: 
                section = {
                    'texts': res[i],
                    'files': idx[i],
                    'conflict': conflict,
                }
                full_docs.append(section)
                full_docs_df = pd.DataFrame(full_docs)

        st.table(df_similar)
        st.write(" ")
    
        st.write('**Contradicting Texts**')
        if full_docs:
            st.table(full_docs_df)
        else:
            st.write('No contradictions found')
        
    st.write(" ")

    with st.expander("Cluster Details"):
        st.write('hello')
    with st.expander("Cluster Details"):
        st.write('hello')

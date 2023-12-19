import streamlit as st 
import pandas as pd
import altair as alt
import transformers as ts
from textblob import TextBlob
import torch
from transformers import pipeline
import evaluate

pipe = pipeline("text-classification", model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa")

def convert_to_df(sentiment2):
    sentiment_dict = {'polarity':sentiment2.polarity,'subjectivity':sentiment2.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
    return sentiment_df

def analyze_token_sentiment(docx):
    regard = evaluate.load("regard")
    pos_list = []
    neg_list = []
    neu_list = []
    res = regard.compute(data = docx)
    for d in res['regard']:
        for l in d:
            continue
    if l['label'] == 'positive' and l['score']>=0.8:
        pos_list.append(d)
        pos_list.append(res)
    elif l['label'] == 'negative' and l['score']>=0.8:
        neg_list.append(d)
        neg_list.append(res)
    else:
        neu_list.append(d)
        result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
    return result


def main():
    st.title("Analisis Sentimen Bahasa Indonesia")
    with st.form("NLPForm"):
        raw_text = st.text_area("Masukkan Teks")
        submit_button = st.form_submit_button(label='Analisis')
    # layout
        col1,col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Hasil")
                sentiment = pipe(raw_text)
                sentiment2 = TextBlob(raw_text).sentiment                
                st.write(sentiment)
                # Emoji

                for teks in sentiment:
                    if teks['label'] == 'Positive':
                        st.markdown("Sentimen :: Positif :smiley: ")
                    elif teks['label'] == 'Negative':
                        st.markdown("Sentimen :: Negatif :angry: ")
                    else:
                        st.markdown("Sentiment:: Netral ")

                if sentiment2.polarity > 0:
                    teks['label'] == 'Positive'
                elif sentiment2.polarity < 0:
                    teks['label'] == 'Negative'
                else:
                    st.markdown(" ")
                                        # Dataframe
                result_df = convert_to_df(sentiment2)
                st.dataframe(result_df)
                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c,use_container_width=True)
            with col2:
                st.info("Token Sentimen")
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)
if __name__ == '__main__':
    main()
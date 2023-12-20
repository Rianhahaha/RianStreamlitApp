import streamlit as st
import wikipediaapi
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import string
from nltk.corpus import stopwords
import time
# Ensure NLTK resources are available
nltk.download('vader_lexicon')
nltk.download('stopwords')

# css 


            

st.set_page_config(
    page_title="Analisis Sentimen Topik di Wikipedia",
    layout="wide"
)

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

# Function to fetch Wikipedia data
wiki_wiki = wikipediaapi.Wikipedia('Gecko/1.0')  # Replace "your-user-agent" with a specific user agent
def fetch_wikipedia_data(query):
    if not query:
        return None
    
    page_py = wiki_wiki.page(query)
    
    if not page_py.exists():
        return None
    else:
        return page_py.text  # Displaying only the first 500 characters for brevity


    
def convert_to_df(sentiment):
    sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
    return sentiment_df

def analyze_token_sentiment(docx):
    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Get the list of English stop words
    stop_words = set(stopwords.words('english'))

    pos_set = set()
    neg_set = set()
    neu_set = set()

    # Tokenize the text using nltk
    tokens = nltk.word_tokenize(docx)

    for i in tokens:
        cleaned_token = i.lower().replace(',', '').replace('.', '')
        cleaned_token = cleaned_token.translate(str.maketrans('', '', string.punctuation + string.digits))

        # Check if the cleaned token is not a stop word
        if cleaned_token not in stop_words:
            res = analyzer.polarity_scores(cleaned_token)['compound']
            if res > 0.1:
                pos_set.add((cleaned_token, res))
            elif res < -0.1:
                neg_set.add((cleaned_token, res))
            else:
                neu_set.add(cleaned_token)

    # Convert sets to lists to maintain order
    pos_list = list(pos_set)
    neg_list = list(neg_set)
    neu_list = list(neu_set)

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result


# Streamlit app
def main():
        st.markdown("""
        <style>
            body {
            font-family: "Source Sans Pro", sans-serif;
            }
        .custom-grid-right, .custom-grid-left {
        position: relative;
        display: flex;
        /* flex-direction: column-reverse; */
        }
        @media screen and (max-width: 1000px) {
        .custom-grid-right {
            flex-direction: column-reverse;
        }
        .custom-grid-left {
            flex-direction: column;
        }
        }
        .custom-grid-item {
        position: relative;
        padding: 1rem;
        }
        .about_us-title {
        font-size: 1.8rem;
        position: relative;
        }
        .about_us-text {
        margin-right: 0;
        }
        .text-justify {
            text-align:justify;
        }
            .css-1kyxreq {
            justify-content:center!important;
            }
        a.btn, .btn {
            text-decoration:none;
            padding:1rem;
            border-radius:20px;
            border:solid 1px #79AF9E;
            background-color: #79AF9E;
            color:white;
            transition: all 1s:

        }
        a.btn .btn:hover {
            background-color:white ;
            color:#79AF9E;
        }
    </style>
                        
                """
                ,unsafe_allow_html=True
            )

        
        selected = option_menu(None, ["Beranda", "Sentimen Analisis", "Tentang"], 
                                icons=['house', 'file-bar-graph', "file-earmark-person"], 
                                menu_icon="cast", default_index=0, orientation="horizontal")
        
        if selected == "Tentang":
            home5,home6 = st.columns(2)
            with home5:
                st.image('Rian.png', caption='About me', width=400)
            with home6:
                    st.markdown("""
    <div class="custom-grid-right round">
    <div class="custom-grid-item">
        <h2 class="about_us-title">
                                Hi! Rian Here!
                                </h2>
        <p class="about_us-text text-justify">

Seorang mahasiswa Teknologi Informasi yang mengkhususkan diri dalam desain grafis, ilustrasi, dan pengembangan web front-end. Keahliannya dalam bidang-bidang ini telah ditunjukkan melalui berbagai proyek yang diselesaikan untuk klien-klien beragam. Dia memiliki minat yang besar dalam desain visual dan implementasinya, termasuk pengembangan situs web front-end, ilustrasi, dan desain grafis.                                
                                </p><br>
                                <a class="btn" href="https://rian-portofolio.vercel.app/index.html">Lihat Portfolio!</a>
    </div>
    </div>
                                    
                            """
                            ,unsafe_allow_html=True
                            )

        if selected == "Beranda":
  
    
            home1,home2 = st.columns(2)
            with home1:
                 st.markdown("""
                    <div class="custom-grid-right round">
                        <div class="custom-grid-item">
                        <h2 class="about_us-title">
                                                Analisis Sentimen
                                                </h2>
                        <p class="about_us-text text-justify">
                            Analisis sentimen adalah proses menganalisis teks digital untuk menentukan apakah nada emosional pesan tersebut positif, negatif, atau netral. Saat ini, perusahaan memiliki data teks dalam volume besar seperti email, transkrip obrolan dukungan pelanggan, komentar media sosial, dan ulasan. Alat analisis sentimen dapat memindai teks ini untuk secara otomatis menentukan sikap penulis terhadap suatu topik. Perusahaan menggunakan wawasan dari analisis sentimen untuk meningkatkan mutu layanan pelanggan dan meningkatkan reputasi merek. 
                        </p>
                        </div>
                    </div>
                                    
                            """
                            ,unsafe_allow_html=True
                        )
            with home2:
                st.image('home.jpg', caption='Analisis Sentimen')
            home3,home4 = st.columns(2)
            with home3:
                st.image('home2.jpg', caption='Wikipedia')
            with home4:
                 st.markdown("""
<div class="custom-grid-right round">
    <div class="custom-grid-item">
      <h2 class="about_us-title">
                             Wikipedia
                             </h2>
      <p class="about_us-text text-justify">Wikipedia adalah ensiklopedia daring yang dapat disunting oleh pengguna. Ini adalah proyek kolaboratif yang memungkinkan pengguna dari seluruh dunia untuk berkontribusi dalam menulis dan mengedit artikel. Wikipedia dikelola oleh Yayasan Wikimedia dan merupakan salah satu sumber daya daring terbesar dan paling populer untuk informasi umum.      </p>
    </div>
  </div>
                                    
                            """
                            ,unsafe_allow_html=True
                        )

                

            

        if selected == "Sentimen Analisis":
            st.title("Analisis Sentimen Data Wikipedia")
            # User input for the Wikipedia page title
            search_query = st.text_input("Masukkan judul Topik Wikipedia: ", placeholder="Masukkan judul Topik Wikipedia")
            wikipedia_data = fetch_wikipedia_data(search_query)
            
            if st.button("Ambil Data"):
                 if wikipedia_data == None:
                    st.warning("Halaman tidak ditemukan. Tolong masukkan judul yang tepat!")
                 else:
                    if search_query:
                    
                        if wikipedia_data:
                            st.info("Data berhasil diambil! ")
                        else:
                           st.warning("Page not found. Please enter a valid Wikipedia page title.")

            tampilkan_data = st.checkbox('Tampilkan Topik Terkait')
            if tampilkan_data:
                if wikipedia_data == None:
                    st.warning("Masukkan Judul terlebih dahulu!!")
                else:
                    st.markdown(f"### Data Wikipedia untuk {search_query}")
                    st.write(wikipedia_data)

            raw_text = wikipedia_data

            submit_button = st.checkbox('Analisis Sentimen')
            
            col1,col2 = st.columns(2)
            if submit_button:
                if raw_text is None:
                    st.warning("Masukkan Judul terlebih dahulu!!")
                else:
                
                    sentiment = TextBlob(raw_text).sentiment
                    # st.write(sentiment)
                    # Emoji
                    # if sentiment.polarity > 0:
                    #     st.markdown("Sentiment:: Positif :smiley: ")
                    # elif sentiment.polarity < 0:
                    #     st.markdown("Sentiment:: Negatif :angry: ")
                    # else:
                    #     st.markdown("Sentiment:: Netral ")
                    with col1:                                      
                        # Dataframe
                        result_df = convert_to_df(sentiment)
                        st.dataframe(result_df)
                        # Visualization
                        c = alt.Chart(result_df).mark_bar().encode(
                            x='metric',
                            y='value',
                            color='metric')
                        st.altair_chart(c,use_container_width=True)
                        # positif negatif

                    with col2:
                        def pos_neg_df(sentiment_pos_neg):

                            sentiment_dict_pos_neg = {
                                            ' Positive Score':vader_sentiment['pos'],
                                            ' Negative Score':vader_sentiment['neg'],
                                            " Neutral Score:": vader_sentiment['neu'],
                                            "Compound Score:": vader_sentiment['compound']
                                            }
                            sentiment_dict_pos_neg_fd = pd.DataFrame(sentiment_dict_pos_neg.items(),columns=['metric','value'])
                            return sentiment_dict_pos_neg_fd
                        
                        analyzer = SentimentIntensityAnalyzer()
                        vader_sentiment = analyzer.polarity_scores(raw_text)
                        result_df = pos_neg_df(sentiment)
                        st.dataframe(result_df)
                        
                        c = alt.Chart(result_df).mark_bar().encode(
                            x='metric',
                            y='value',
                             color=alt.Color('metric', scale=alt.Scale(range=['#FF5733', '#3498db', '#27ae60', '#FFC300']))  # Specify your custom colors here
                            )
                        st.altair_chart(c,use_container_width=True)

                        if vader_sentiment['pos'] > vader_sentiment['neg']:
                            st.markdown("Sentiment:: Positif :smiley: ")
                        elif vader_sentiment['pos'] < vader_sentiment['neg']:
                            st.markdown("Sentiment:: Negatif :angry: ")
                        else:
                            st.markdown("Sentiment:: Netral ")
                    

                
                    st.info("Token Sentimen")
                    token_sentiments = analyze_token_sentiment(raw_text)
                    st.write(token_sentiments)

                    

                        # Bar chart for VADER positive and negative sentiments
                        # bar_chart_data = {'Sentiment': ['Positive', 'Negative'], 'Score': [vader_sentiment['pos'], vader_sentiment['neg']]}
                        # bar_chart_df = pd.DataFrame(bar_chart_data)
                        
                        # st.bar_chart(bar_chart_df.set_index('Sentiment'))
                # VADER sentiment analysis

        
                    
                    def prepare_dataframe(sentiments):
                        # Extract positive, negative, and neutral tokens along with scores
                        positive_tokens = [token_info[0] for token_info in sentiments['positives']]
                        positive_scores = [token_info[1] for token_info in sentiments['positives']]
                        
                        negative_tokens = [token_info[0] for token_info in sentiments['negatives']]
                        negative_scores = [token_info[1] for token_info in sentiments['negatives']]
                        
                        neutral_tokens = [token for token in sentiments['neutral']]
                        neutral_scores = [None] * len(neutral_tokens)

                        # Find the length of the longest list
                        max_length = max(len(positive_tokens), len(negative_tokens), len(neutral_tokens))

                        # Extend lists with None values to make them equal in length
                        positive_tokens += [None] * (max_length - len(positive_tokens))
                        positive_scores += [None] * (max_length - len(positive_scores))
                        
                        negative_tokens += [None] * (max_length - len(negative_tokens))
                        negative_scores += [None] * (max_length - len(negative_scores))
                        
                        neutral_tokens += [None] * (max_length - len(neutral_tokens))
                        neutral_scores += [None] * (max_length - len(neutral_scores))

                        # Create DataFrame
                        token_sentiments_df = pd.DataFrame({
                            'Positive Tokens': positive_tokens,
                            'Positive Scores': positive_scores,
                            'Negative Tokens': negative_tokens,
                            'Negative Scores': negative_scores,
                            'Neutral Tokens': neutral_tokens,
                            'Neutral Scores': neutral_scores
                        })

                        return token_sentiments_df
                    
# Display positive tokens and their scores
                 # Create DataFrames for positive, negative, and neutral tokens
                    # positive_df = pd.DataFrame(token_sentiments['positives'], columns=['Token', 'Score'])
                    # negative_df = pd.DataFrame(token_sentiments['negatives'], columns=['Token', 'Score'])
                    # neutral_df = pd.DataFrame({'Token': token_sentiments['neutral'], 'Score': [None] * len(token_sentiments['neutral'])})

                    # # Add a new column to indicate sentiment
                    # positive_df['Sentiment'] = 'Positive'
                    # negative_df['Sentiment'] = 'Negative'
                    # neutral_df['Sentiment'] = 'Neutral'

                    # # Combine the DataFrames into a single DataFrame
                    # combined_df = pd.concat([positive_df, negative_df, neutral_df])

                    # # Display the combined DataFrame
                    # st.write("Combined Token Sentiments:")
                    # st.dataframe(combined_df)
                        

                    # Display the DataFrame using Streamlit     
                    token_sentiments_df = prepare_dataframe(token_sentiments)
                    st.dataframe(token_sentiments_df)

 

                    token_sentiments_df = prepare_dataframe(token_sentiments)

                    # Combine tokens and scores into a single column (feature)
                    token_sentiments_df['Features'] = token_sentiments_df.apply(
                        lambda row: f"{row['Positive Tokens']} {row['Negative Tokens']} {row['Neutral Tokens']}", axis=1
                    )
                   

                    # Define a threshold to convert scores to labels
                    threshold = 0.0  # You may adjust this threshold based on your data characteristics

                    # Assign labels based on the threshold
                    token_sentiments_df['Label'] = token_sentiments_df.apply(
                        lambda row: 'positive' if row['Positive Scores'] > threshold else ('negative' if row['Negative Scores'] else 'neutral'),
                        axis=1
                    )
                    # col1,col2 = st.columns(2)
                    # with col1:
                    #     token_sentiments_df['Features']
                    # with col2:
                    #     token_sentiments_df['Label']

                    # Label encoding for 'Label' column
                    label_encoder = LabelEncoder()
                    token_sentiments_df['Label'] = label_encoder.fit_transform(token_sentiments_df['Label'])
                  
                    X_train, X_test, y_train, y_test = train_test_split(
                        token_sentiments_df['Features'], token_sentiments_df['Label'], test_size=0.2, random_state=42
                    )
                    st.write("Data Uji :", X_test.shape[0])
                    st.write("Data Latih :", X_train.shape[0])
                    

                    # Convert the text data into features (bag-of-words representation)
                    vectorizer = CountVectorizer()
                    X_train_vectorized = vectorizer.fit_transform(X_train)
                    X_test_vectorized = vectorizer.transform(X_test)
               

                    # Train the Naive Bayes model
                    nb_classifier = MultinomialNB()
                    nb_classifier.fit(X_train_vectorized, y_train)

                    # Make predictions
                    predictions = nb_classifier.predict(X_test_vectorized)

                    # Decode labels back to original values for evaluation
                    y_test = label_encoder.inverse_transform(y_test)
                    predictions = label_encoder.inverse_transform(predictions)
                
                   

                    # Evaluate the model
                    accuracy = metrics.accuracy_score(y_test, predictions)
                    precision = metrics.precision_score(y_test, predictions, average='weighted', zero_division=1)
                    recall = metrics.recall_score(y_test, predictions, average='weighted')
                    f1 = metrics.f1_score(y_test, predictions, average='weighted')

                    # st.write(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
                    metrics_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                        'Value': [accuracy, precision, recall, f1]
                    }
                    metrics_df = pd.DataFrame(metrics_data)

                    # Display the DataFrame using Streamlit
                    st.dataframe(metrics_df)



            related_value = st.number_input('Masukkan Jumlah related topic yang ingin dilihat', step=1, max_value=101)
        
            def fetch_related_topics(query, related_value):
                page_py = wiki_wiki.page(query)
                return [link.title for link in page_py.links.values()][:related_value] 
            
            if st.button("Tampilkan Topik terkait"):
                if raw_text == None:
                   st.warning("Masukkan Judul terlebih dahulu!!")
                elif related_value == 0:
                    st.warning("Isi jumlah Topik terkait terlebih dahulu!")
                else:
                    if raw_text:
                        related_topics = fetch_related_topics(search_query, related_value)
                        st.write("Related Topics:")
                        topics_df = pd.DataFrame({"Topik terkait": related_topics})
                        st.dataframe(topics_df)


                        wordcloud_fig = generate_wordcloud(" ".join(related_topics))
                        st.pyplot(wordcloud_fig)
       

                
                    
              







if __name__ == "__main__":
    main()

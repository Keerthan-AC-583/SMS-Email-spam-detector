import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ---------------- NLTK ----------------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()


# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)


# ---------------- Simple CSS ----------------
st.markdown("""
<style>

    .stApp {
        background-color: #f5f7fb;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #222222;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        margin-bottom: 30px;
    }

    .result {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
        margin-top: 20px;
    }

    .spam {
        background-color: #ffe5e5;
        color: #d00000;
    }

    .safe {
        background-color: #e5f8e9;
        color: #16803c;
    }

</style>
""", unsafe_allow_html=True)


# ---------------- Load Model ----------------
@st.cache_resource
def load_model():

    with open("tfidf.pkl", "rb") as file:
        tfidf = pickle.load(file)

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    return tfidf, model


tfidf, model = load_model()


# ---------------- Text Preprocessing ----------------
def transform_text(text):

    text = text.lower()

    tokens = nltk.word_tokenize(text)

    # Remove special characters
    tokens = [
        word for word in tokens
        if word.isalnum()
    ]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Stemming
    tokens = [
        ps.stem(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ---------------- Website ----------------

st.markdown(
    '<div class="title">📱 SMS Spam Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter a message to check whether it is Spam or Not Spam'
    '</div>',
    unsafe_allow_html=True
)


# Message input
message = st.text_area(
    "Enter your message",
    height=150,
    placeholder="Example: Congratulations! You have won a prize..."
)


# Predict button
if st.button("🔍 Predict", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Preprocess
        transformed_message = transform_text(message)

        # TF-IDF
        vector = tfidf.transform([transformed_message])

        # Prediction
        result = model.predict(vector)[0]

        # Display result
        if result == 1 or result == "spam":

            st.markdown(
                '<div class="result spam">'
                '🚨 Spam Detected'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result safe">'
                '✅ Not Spam'
                '</div>',
                unsafe_allow_html=True
            )


# ---------------- Footer ----------------

st.markdown("---")

st.caption(
    "Built using Python • NLP • TF-IDF • Machine Learning • Streamlit"
)
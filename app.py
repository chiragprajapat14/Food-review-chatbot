
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# 1. Page Configuration & Title
st.set_page_config(page_title="Swiggy/Zomato Sentiment Chatbot", page_icon="🍕", layout="centered")
st.title("🍕 Swiggy & Zomato Review Chatbot")
st.write("Chirag Prajapat's Deep Learning Project - Powered by LSTM")

# 2. Model aur Tokenizer ko load karna
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('zomato_sentiment_lstm.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

try:
    model, tokenizer = load_assets()
    st.success("🤖 AI Model Loaded Successfully!")
except Exception as e:
    st.error("Error loading model files. Make sure .h5 and .pickle files are in the same folder.")

# 3. User Interface (Chat Input)
user_review = st.text_input("Enter a food review (Hinglish/English):", placeholder="e.g., Khaana bohot swadisht tha par delivery late thi...")

if st.button("Analyze & Reply"):
    if user_review.strip() != "":
        # Preprocessing exactly like Colab training
        seq = tokenizer.texts_to_sequences([user_review])
        padded = pad_sequences(seq, padding="post", maxlen=15, truncating="post")
        
        # Prediction
        prediction = model.predict(padded, verbose=0)[0][0]
        
        # UI Responses based on Sentiment
        st.write("---")
        if prediction >= 0.5:
            st.markdown(f"### 😎 **Positive Sentiment Detected!** (Confidence: {prediction*100:.2f}%)")
            st.chat_message("assistant").write("💬 **Bot Auto-Reply:** Thank you so much for your feedback! We are glad you loved the food and service. Keep ordering! ❤️")
        else:
            st.markdown(f"### 😡 **Negative Sentiment Detected!** (Confidence: {(1-prediction)*100:.2f}%)")
            st.chat_message("assistant").write("💬 **Bot Auto-Reply:** We sincerely apologize for this terrible experience. Your issue has been escalated to our priority support team. 🛠️")
    else:
        st.warning("Please type a review first!")

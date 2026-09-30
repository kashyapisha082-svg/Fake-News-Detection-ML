import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

# Create prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.title("📰 Fake News Detection using Machine Learning")

st.write(
    "Enter a news article below and the machine learning model "
    "will predict whether it is Fake or Real."
)

# Accuracy
st.info("Model Accuracy on Test Data: 98.46%")

# News input
news_text = st.text_area(
    "Enter News Article:",
    height=250,
    placeholder="Paste the news article here..."
)

# Check button
if st.button("🔍 Check News"):

    if news_text.strip() == "":
        st.warning("Please enter a news article first.")

    else:
        # Convert text into numerical features
        news_vector = vectorizer.transform([news_text])

        # Prediction
        prediction = model.predict(news_vector)[0]

        if prediction == 0:
            result = "FAKE NEWS"
            st.error("❌ FAKE NEWS")
        else:
            result = "REAL NEWS"
            st.success("✅ REAL NEWS")

        # Save prediction in history
        st.session_state.history.append({
            "News": news_text[:100],
            "Prediction": result
        })

# Prediction history
st.subheader("📋 Prediction History")

if len(st.session_state.history) > 0:
    for item in st.session_state.history:
        st.write("News:", item["News"])
        st.write("Prediction:", item["Prediction"])
        st.divider()
else:
    st.write("No predictions yet.")
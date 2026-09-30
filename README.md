# Fake News Detection using Machine Learning

## Project Description
This project detects whether a news article is Fake or Real using Machine Learning.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

## Dataset
The project uses the ISOT Fake News Dataset, containing fake and real news articles.

## Machine Learning Model
TF-IDF Vectorization is used to convert news text into numerical features.

Logistic Regression is then used to classify the news as:
- Fake News
- Real News

## Model Accuracy
The model achieved 98.46% accuracy on the test data.

## Project Files
- `app.py` - Streamlit web application
- `train_model.py` - Model training code
- `fake_news_model.pkl` - Trained machine learning model
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn streamlit

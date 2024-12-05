from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get text input from user
        user_input = request.form['text']
        
        # Transform the text using the saved vectorizer
        text_vectorized = vectorizer.transform([user_input])
        
        # Make a prediction using the model
        prediction = model.predict(text_vectorized)
        
        # Convert prediction to label (0, 1, 2: Negative, Positive, Neutral)
        sentiment = ['Negative', 'Positive', 'Neutral'][prediction[0]]
        
        return render_template('index.html', prediction_text='Sentiment: {}'.format(sentiment))

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

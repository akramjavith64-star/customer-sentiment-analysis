
from flask import Flask, render_template, request
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import webbrowser
from threading import Timer

# Initialize Flask
app = Flask(__name__)

# Download NLP resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load the latest trained model
model_path = os.path.join("model", "sentiment_model.joblib")
model = joblib.load(model_path)

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
stop_words -= {"not", "no", "nor"}

lemmatizer = WordNetLemmatizer()

# Same cleaning function used during training
def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"\b(isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|couldn't|won't|wouldn't|shouldn't)\b",
        " not ",
        text
    )

    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Predict sentiment
@app.route("/predict", methods=["POST"])
def predict():

    text = request.form.get("feedback", "").strip()

    if not text:
        return render_template(
            "index.html",
            error="Please enter a customer review."
        )

    cleaned_text = clean_text(text)

    prediction = model.predict([cleaned_text])[0]

    return render_template(
        "index.html",
        prediction=prediction,
        feedback=text
    )


# Run Flask
if __name__ == "__main__":
    Timer(
        1,
        lambda: webbrowser.open_new("http://127.0.0.1:5000")
    ).start()

    app.run(debug=False, use_reloader=False)
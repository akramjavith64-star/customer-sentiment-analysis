
import pandas as pd
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load datasets
df1 = pd.read_csv("data/customer_sentiment_training.csv")
df2 = pd.read_csv("data/customer_sentiment_more_text.csv")

# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)

# Clean dataset
df = df.dropna(subset=["text", "label"])
df = df.drop_duplicates()

# Standardize labels
df["label"] = df["label"].astype(str).str.strip().str.lower()

print("Total dataset:", df.shape)
print("\nSentiment distribution:")
print(df["label"].value_counts())

# Initialize NLP tools
stop_words = set(stopwords.words("english"))

# Preserve important negative words
stop_words -= {"not", "no", "nor"}

lemmatizer = WordNetLemmatizer()

# Text cleaning
def clean_text(text):
    text = str(text).lower()

    # Preserve negation in common contractions
    text = re.sub(r"\b(isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|couldn't|won't|wouldn't|shouldn't)\b", " not ", text)

    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Clean reviews
df["text"] = df["text"].apply(clean_text)

# Remove empty reviews
df = df[df["text"].str.strip() != ""]

# Features and labels
X = df["text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train model
print("\nTraining model...")

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model in model folder
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/sentiment_model.joblib")

print("\nModel saved successfully!")

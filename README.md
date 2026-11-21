<<<<<<< HEAD
# Sentiment Analysis — Customer Feedback


## Setup
1. Create a virtual env: `python -m venv venv` and `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
2. Install: `pip install -r requirements.txt`
3. Train model: `python train_model.py` (this saves `model/sentiment_model.joblib`)
4. Run app: `python app.py` or `gunicorn app:app` for production
5. Open http://localhost:5000


## Deploy
- Use Heroku, Render, Railway, or any service that supports Python/Flask.
- For Heroku: `heroku create`, `git push heroku main` and set buildpacks if required.
=======
# customer-sentiment-analysis
Customer Sentiment Analysis using Python, Machine Learning, NLP and Flask to classify customer reviews into positive, negative and neutral sentiments.
>>>>>>> fca45d8de4c8222ecb53c505fde5f50e45a70fee

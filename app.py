import pandas as pd
import pickle
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load spaCy model and NLTK stopwords
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stopwords_set = set(stopwords.words("english"))

# Function required by the pipeline's FunctionTransformer
def tokens_to_string(token_series):
    return token_series.apply(lambda x: " ".join(x) if isinstance(x, list) and x else "")

# Load the saved pipeline (TF-IDF with Linear SVM)
try:
    with open(r"C:\Users\lenovo\Downloads\trainingandtestdata\sentiment_model_tfidf_svm.pkl", "rb") as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    print("Error: Pipeline file not found at specified path.")
    exit(1)
except Exception as e:
    print(f"Error loading pipeline: {str(e)}")
    exit(1)

# Preprocessing functions (matching notebook)
def clean(text):
    text = re.sub(r"http://\w*\.\w*/?\w?/?\w*|http://\w*-\w*.\w*.\w*|http:/\W*", "", text)
    text = re.sub(r"@\w*|#\w*", "", text)
    text = re.sub(r'[^\x00-\x7F]+', "", text)
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    return " ".join(text.split())

def case_folding_and_tokenize(text):
    text = text.lower()
    doc = nlp(text)
    return [token.lemma_ for token in doc] if doc else []

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stopwords_set] if tokens else []

def remove_punctuation(tokens):
    # Reprocess the tokens as a Doc to get pos_ attributes
    doc = nlp(" ".join(tokens))
    return [token.text for token in doc if token.pos_ != "PUNCT"] if doc else []

def replace_numbers(tokens):
    return ["NUM" if re.search(r'\d', token) else token for token in tokens] if tokens else []

# Root route to avoid 404 when accessing base URL
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Sentiment140 Twitter API",
        "endpoint": "/predict",
        "method": "POST",
        "example": {"tweet": "I love this movie!"},
        "usage": "Send a POST request to /predict with a JSON body containing a 'tweet' field."
    })

# Endpoint to predict sentiment
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "tweet" not in data:
            return jsonify({"error": "No tweet provided in JSON payload"}), 400

        tweet = data["tweet"]
        if not isinstance(tweet, str) or not tweet.strip():
            return jsonify({"error": "Tweet must be a non-empty string"}), 400

        # Compute numerical features from the tweet
        tweets_length = len(tweet)
        tweets_words_length = len(word_tokenize(tweet))
        tweets_sentences_length = len(sent_tokenize(tweet))
        # Dummy date values (matching dataset default)
        year = 2009
        month = 6
        day = 16

        # Preprocess the tweet to get cleaned_text list
        cleaned = clean(tweet)
        tokenized = case_folding_and_tokenize(cleaned)
        no_stopwords = remove_stopwords(tokenized)
        no_punct = remove_punctuation(no_stopwords)
        final_tokens = replace_numbers(no_punct)

        # Create DataFrame for pipeline input
        input_df = pd.DataFrame({
            "tweets_length": [tweets_length],
            "tweets_words_length": [tweets_words_length],
            "tweets_sentences_length": [tweets_sentences_length],
            "year": [year],
            "month": [month],
            "day": [day],
            "cleaned_text": [final_tokens]
        })

        # Predict using the pipeline
        prediction = pipeline.predict(input_df)[0]
        sentiment = "positive" if prediction == 1 else "negative"

        return jsonify({"tweet": tweet, "sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
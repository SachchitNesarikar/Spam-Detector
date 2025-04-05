import os
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
USER_INPUT_CSV = os.path.join(DATA_DIR, "user_prompts.csv")

SPAM_CSV = os.path.join(DATA_DIR, "spam.csv")
NB_MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes.joblib")
LR_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

if not os.path.exists(SPAM_CSV):
    raise FileNotFoundError(f"CSV file not found at: {SPAM_CSV}")

df = pd.read_csv(SPAM_CSV, encoding="latin1")[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 1, 'spam': 0})
df['clean_text'] = df['text'].apply(preprocess_text)

if os.path.exists(VECTORIZER_PATH):
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["clean_text"])
    joblib.dump(vectorizer, VECTORIZER_PATH)

X = vectorizer.transform(df["clean_text"])
y = df["label"]

def train_and_save_model(model_class, X, y, model_path):
    model = model_class()
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

nb_model = joblib.load(NB_MODEL_PATH) if os.path.exists(NB_MODEL_PATH) else train_and_save_model(MultinomialNB, X, y, NB_MODEL_PATH)
lr_model = joblib.load(LR_MODEL_PATH) if os.path.exists(LR_MODEL_PATH) else train_and_save_model(lambda: LogisticRegression(max_iter=1000), X, y, LR_MODEL_PATH)

def predict_spam(text, model_choice="naive_bayes"):
    if not text.strip():
        return {"error": "Empty text"}

    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])

    if model_choice == "naive_bayes":
        prob = nb_model.predict_proba(vec)[0][1]
    elif model_choice == "logistic_regression":
        prob = lr_model.predict_proba(vec)[0][1]
    else:
        return {"error": f"Invalid model '{model_choice}'"}

    is_spam = prob >= 0.5  # True if spam
    result = {
        "input": text,
        "cleaned": clean,
        "model": model_choice,
        "is_spam": bool(is_spam),  # native bool
        "spam_probability": float(round(prob, 4))  # native float
    }
    return result

'''def append_to_csv(text, is_spam, model, prob):
    row = pd.DataFrame([{
        "text": text,
        "is_spam": is_spam,
        "model": model,
        "probability": prob
    }])

    if os.path.exists(USER_INPUT_CSV):
        row.to_csv(USER_INPUT_CSV, mode='a', index=False, header=False)
    else:
        row.to_csv(USER_INPUT_CSV, mode='w', index=False, header=True) '''

def retrain_model(X_train, y_train, model_choice="naive_bayes"):
    global nb_model, lr_model

    if model_choice == "naive_bayes":
        model = MultinomialNB()
        path = NB_MODEL_PATH
    else:
        model = LogisticRegression(max_iter=1000)
        path = LR_MODEL_PATH

    model.fit(vectorizer.transform(X_train), y_train)
    joblib.dump(model, path)

    if model_choice == "naive_bayes":
        nb_model = model
    else:
        lr_model = model

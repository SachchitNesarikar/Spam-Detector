import joblib
import os

MODEL_PATH = "ml_algos/models/"

def load_model(file_name):
    file_path = os.path.join(MODEL_PATH, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: {file_name} not found!")
    return joblib.load(file_path)

vectorizer = load_model("vectorizer.joblib")
naive_bayes_model = load_model("naive_bayes.joblib")
logistic_regression_model = load_model("logistic_regression.joblib")

def predict_spam(text, model_choice="naive_bayes"):
    if not text.strip():
        return -1.0

    processed_text = text.lower()
    text_vector = vectorizer.transform([processed_text])

    if model_choice == "naive_bayes":
        model = naive_bayes_model
    elif model_choice == "logistic_regression":
        model = logistic_regression_model
    else:
        return -2.0

    spam_proba = model.predict_proba(text_vector)[0][1]
    return round(spam_proba * 100, 2)

print("✅ Models and vectorizer loaded successfully!")
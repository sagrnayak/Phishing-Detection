# Import required libraries
import pandas as pd
import re

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# -------------------- LOAD DATA --------------------
def load_data():
    """
    Loads dataset from CSV file.
    """
    data = pd.read_csv("emails.csv")
    return data


# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    """
    Cleans email text:
    - Removes URLs
    - Removes special characters
    - Converts to lowercase
    """
    text = re.sub(r"http\S+", "", text)   # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove special chars
    text = text.lower()
    return text


# -------------------- FEATURE EXTRACTION --------------------
def extract_features(data):
    """
    Converts text into numerical features using TF-IDF.
    """
    # Apply cleaning
    data['text'] = data['text'].apply(clean_text)

    # Convert text to vectors
    vectorizer = TfidfVectorizer(stop_words='english')

    X = vectorizer.fit_transform(data['text'])  # Features
    y = data['label']                           # Labels

    return X, y, vectorizer


# -------------------- MODEL TRAINING --------------------
def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


# -------------------- MODEL EVALUATION --------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print("\n📊 Accuracy:", accuracy)
    print("\n📉 Confusion Matrix:\n", cm)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    # Step 1: Load dataset
    data = load_data()

    # Step 2: Extract features
    X, y, vectorizer = extract_features(data)

    # Step 3: Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Train model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate model
    evaluate_model(model, X_test, y_test)

    # Step 6: Test custom email
    test_email = input("\nEnter an email to check: ")

    # Transform input using same vectorizer
    test_email_clean = clean_text(test_email)
    test_vector = vectorizer.transform([test_email_clean])

    prediction = model.predict(test_vector)

    print("\n🔍 Prediction:", prediction[0])
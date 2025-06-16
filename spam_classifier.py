import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df = pd.read_csv('spam_dataset.csv')
df['cleaned'] = df['text'].apply(clean_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['cleaned']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
nb_model.fit(X_train, y_train)

svm_model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])
svm_model.fit(X_train, y_train)

def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"{name} Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("-" * 30)

evaluate(nb_model, "Naive Bayes")
evaluate(svm_model, "SVM")

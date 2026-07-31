import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = pd.read_csv("spam.csv")

print(data.head())
X = data["Message"]
y = data["Label"]
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = MultinomialNB()

model.fit(X_train, y_train)
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy)
email = ["Congratulations! You have won a free laptop."]

email_vector = vectorizer.transform(email)

result = model.predict(email_vector)

print("Prediction:", result[0])
email = ["Please attend the class at 9 AM tomorrow."]
email_vector = vectorizer.transform(email)
result = model.predict(email_vector)
print("Prediction:", result[0])

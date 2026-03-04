import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("spam.csv")

X = data["message"]
y = data["label"]

vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vector, y)

msg = input("Enter a message: ")

msg_vector = vectorizer.transform([msg])

prediction = model.predict(msg_vector)

if prediction[0] == "spam":
    print("Prediction: SPAM ")
else:
    print("Prediction: NOT SPAM ")
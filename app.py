import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


st.title("📩 Spam Detection AI")

st.write("Enter a message to check if it is Spam or Not Spam")


data = pd.read_csv("spam.csv")

X = data["message"]
y = data["label"]


vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)


model = MultinomialNB()
model.fit(X_vector, y)


msg = st.text_input("Enter message:")

if st.button("Check Message"):

    msg_vector = vectorizer.transform([msg])

    prediction = model.predict(msg_vector)
    probability = model.predict_proba(msg_vector)

    confidence = max(probability[0]) * 100

    if prediction[0] == "spam":
        st.error(f"⚠️ This message is SPAM ({confidence:.2f}% confidence)")
    else:
        st.success(f"✅ This message is NOT SPAM ({confidence:.2f}% confidence)")
# 📩 Spam Detection AI

A Machine Learning web application that detects whether a message is **Spam** or **Not Spam** using Natural Language Processing and a Naive Bayes classifier.

---

## 🚀 Features

* Detects spam messages instantly
* Uses Machine Learning for classification
* Displays prediction confidence
* Simple interactive web interface
* Trained on a real-world SMS spam dataset

---

## 🧠 Machine Learning Model

This project uses a **Multinomial Naive Bayes classifier** for text classification.

Pipeline used in this project:

Dataset → Text Vectorization → Machine Learning Model → Prediction

Steps:

1. Messages are converted into numerical vectors using **CountVectorizer**
2. The model learns patterns from labeled spam and non-spam messages
3. The trained model predicts whether a new message is spam

---

## 📊 Dataset

The model is trained on the **SMS Spam Collection Dataset**, containing over **5,500 SMS messages** labeled as spam or not spam.

Example data:

```
ham   Hey are we still meeting today?
spam  Congratulations! You have won a free ticket
ham   Call me when you reach home
spam  URGENT! Claim your reward now
```

---

## 🖥️ Web Application

The application is built using **Streamlit**, allowing users to test messages directly in the browser.

Example:

Input:

```
Win a free iPhone now
```

Output:

```
⚠️ This message is SPAM (95% confidence)
```

---

## ⚙️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* Natural Language Processing

---

## 📂 Project Structure

```
spam_detection
│
├── app.py
├── train_model.py
├── spam.csv
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Project

Clone the repository:

```
git clone https://github.com/rohanelias/spam_detection.git
```

Navigate to the project folder:

```
cd spam-detection-ai
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the web app:

```
python -m streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🎯 Future Improvements

* Improve accuracy with TF-IDF vectorization
* Add deep learning model
* Support email spam detection
* Deploy as a public web service

---

## 📌 Author

Developed as a Machine Learning project demonstrating spam detection using NLP techniques.

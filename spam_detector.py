# Spam Message Classifier with Simple UI

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import tkinter as tk

# -----------------------
# Load dataset (same folder)
# -----------------------
data = pd.read_csv("spam.csv", encoding='latin-1')

# take only required columns
data = data.iloc[:, :2]
data.columns = ['label', 'message']

# convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# remove empty rows
data.dropna(inplace=True)

# -----------------------
# Train model
# -----------------------
cv = CountVectorizer()
X = cv.fit_transform(data['message'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# check accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# -----------------------
# Function for prediction
# -----------------------
def check_message():
    msg = entry.get()
    
    if msg.strip() == "":
        result_label.config(text="Enter a message first")
        return
    
    msg_data = cv.transform([msg])
    result = model.predict(msg_data)[0]
    
    if result == 1:
        result_label.config(text="Spam ❌")
    else:
        result_label.config(text="Not Spam ✅")

# -----------------------
# UI Design
# -----------------------
root = tk.Tk()
root.title("Spam Checker")

title = tk.Label(root, text="Spam Message Checker", font=("Arial", 14))
title.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

btn = tk.Button(root, text="Check", command=check_message)
btn.pack(pady=5)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
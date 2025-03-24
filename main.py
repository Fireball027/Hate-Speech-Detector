# Import libraries
import pandas as pd
import numpy as np
import re
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import shap

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the data
data = pd.read_csv("twitter_data.csv")

# Mapping class labels
data['labels'] = data['class'].map({0: "Hate speech", 1: "Not offensive", 2: "Neutral"})

# Select relevant columns
data = data[["tweet", "labels"]]

# Preprocess text
stopword = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)    # Remove URLs
    text = re.sub(r'\@w+|\#', '', text)    # Remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)    # Remove special characters
    tokens = nltk.word_tokenize(text)    # Tokenization
    tokens = [word for word in tokens if word not in stopword]    # Remove stopwords
    return " ".join(tokens)


# Apply text cleaning
data["cleaned_tweet"] = data["tweet"].apply(clean_text)

# Tokenization
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(data["cleaned_tweet"])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(data["cleaned_tweet"])
max_len = 100    # Maximum tweet length
X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

# Convert labels to numeric values
label_map = {"Hate speech": 0, "Not offensive": 1, "Neutral": 2}
y = data["labels"].map(label_map).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights to handle imbalances
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Build LSTM (Long Short-Term Memory) model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation="softmax")
])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test),
                    class_weight=class_weights)

# Predictions
y_pred = model.predict(X_test)
y_pred_best = np.argmax(y_pred, axis=1)

# Classification report
print(classification_report(y_test, y_pred_best, target_names=label_map.keys()))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_best)
print(f"Model Accuracy: {accuracy:.4f}")

# Visualization
# Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=data['labels'], palette="coolwarm")
plt.title("Class Distribution of Tweets")
plt.xlabel("Labels")
plt.ylabel("Count")
plt.show()

# Word Cloud
for label in data['labels'].unique():
    subset = data[data['labels'] == label]
    text = " ".join(subset['cleaned_tweet'])

    plt.figure(figsize=(6, 4))
    wordcloud = WordCloud(width=600, height=400, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {label} Tweets")
    plt.show()

# Training Loss & Accuracy Plot
plt.figure(figsize=(10, 4))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")

plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_map.keys())
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix for Tweet Classification")
plt.show()

# SHAP (SHapley Additive exPlanations) Explanation
explainer = shap.Explainer(lambda x: model.predict(x, batch_size=32), X_train[:100])
shap_values = explainer(X_test[:10])

shap.summary_plot(shap_values, X_test[:10], feature_names=tokenizer.word_index)


# User Inputs
def predict_tweet(tweet):
    tweet_cleaned = clean_text(tweet)
    tweet_seq = tokenizer.texts_to_sequences([tweet_cleaned])
    tweet_padded = pad_sequences(tweet_seq, maxlen=max_len, padding="post", truncating="post")

    pred = model.predict(tweet_padded)
    class_index = np.argmax(pred)
    predicted_label = [k for k, v in label_map.items() if v == class_index][0]

    print(f"The tweet is classified as: {predicted_label}")


# Take user input and predict
tweet = input("Enter your tweet: ")
predict_tweet(tweet)

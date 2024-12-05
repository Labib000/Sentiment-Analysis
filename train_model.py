import pandas as pd

# Load the text and labels files
train_texts = open('datasets/sentiment/train_text.txt', 'r').readlines()
train_labels = open('datasets/sentiment/train_labels.txt', 'r').readlines()

val_texts = open('datasets/sentiment/val_text.txt', 'r').readlines()
val_labels = open('datasets/sentiment/val_labels.txt', 'r').readlines()

# Check the length of text and labels to ensure they match
print(f"Training texts count: {len(train_texts)}")
print(f"Training labels count: {len(train_labels)}")
print(f"Validation texts count: {len(val_texts)}")
print(f"Validation labels count: {len(val_labels)}")

# Combine text and labels into a DataFrame
train_data = pd.DataFrame({
    'text': train_texts,
    'label': [int(label.strip()) for label in train_labels]
})

val_data = pd.DataFrame({
    'text': val_texts,
    'label': [int(label.strip()) for label in val_labels]
})

# Remove any empty rows or rows where text is just whitespace
train_data = train_data[train_data['text'].str.strip() != '']
val_data = val_data[val_data['text'].str.strip() != '']

# Check for any missing values
print(f"Missing values in train data: {train_data.isnull().sum()}")
print(f"Missing values in val data: {val_data.isnull().sum()}")

# Clean text data if necessary (optional step)
train_data['text'] = train_data['text'].str.strip()
val_data['text'] = val_data['text'].str.strip()

# Check the distribution of labels
print("Train label distribution:")
print(train_data['label'].value_counts())

print("Validation label distribution:")
print(val_data['label'].value_counts())

# Saving cleaned data (optional)
train_data.to_csv('datasets/sentiment/cleaned_train.csv', index=False)
val_data.to_csv('datasets/sentiment/cleaned_val.csv', index=False)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Vectorize the text
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
X_val = vectorizer.transform(val_data['text'])

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, train_data['label'])

# Make predictions on the validation data
y_pred = model.predict(X_val)

# Evaluate the model
print("Classification Report:")
print(classification_report(val_data['label'], y_pred))


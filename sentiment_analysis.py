# sentiment_analysis.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load your dataset
df = pd.read_csv("customer_reviews_large.csv")





# Step 2: Preprocess
df.dropna(inplace=True)
df['Review'] = df['Review'].astype(str)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Prediction and Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

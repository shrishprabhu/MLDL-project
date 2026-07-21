import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#loading dataset

DATA_PATH = "dataset/Phones_accelerometer.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")
print(df.head())

# removing and keeping only required columns

df = df[['x', 'y', 'z', 'gt']]

# Remove missing values
df = df.dropna()

# Use a smaller sample for faster training
df = df.sample(n=50000, random_state=42)

print("Using sample size:", len(df))
print("\nDataset Shape:", df.shape)

#encoding labels

encoder = LabelEncoder()

df['gt'] = encoder.fit_transform(df['gt'])

# Features & Labels
X = df[['x', 'y', 'z']]
y = df['gt']

#train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# training the model

model = RandomForestClassifier(
    n_estimators=5,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

#evaluating the model

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# saving the model and encoder

os.makedirs("backend", exist_ok=True)

joblib.dump(model, "backend/model.pkl")
joblib.dump(encoder, "backend/encoder.pkl")

print("\nModel saved successfully!")
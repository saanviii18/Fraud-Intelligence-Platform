import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.metrics import classification_report


# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Features
X = df.drop("Class", axis=1)

# Target
y = df["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = XGBClassifier()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Results
print(classification_report(y_test, predictions))
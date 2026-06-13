import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier()

model.fit(X_train, y_train)

joblib.dump(model, "fraud_model.pkl")

print("Model saved successfully!")
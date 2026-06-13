import pandas as pd
import shap
import joblib

model = joblib.load("fraud_model.pkl")

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)

explainer = shap.TreeExplainer(model)

sample = X.iloc[:100]

shap_values = explainer.shap_values(sample)

shap.summary_plot(
    shap_values,
    sample
)

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import base64
import matplotlib.pyplot as plt
from io import BytesIO

app = FastAPI()

# Load model and SHAP explainer
model = joblib.load("loan_model_stdsc_xgb_pipeline.pkl")
explainer = joblib.load("shap_explainer.pkl")

class InputData(BaseModel):
    Age: int
    Experience: int
    Income: float
    ZIPCode: int 
    Family: int
    CCAvg: float
    Education: int
    Mortgage: float
    SecuritiesAccount: int
    CDAccount: int
    Online: int
    CreditCard: int

@app.post("/predict")
def predict(input_data: InputData):
    data_dict = input_data.dict()
    data_dict["ZIP Code"] = data_dict.pop("ZIPCode")
    data_dict["Securities Account"] = data_dict.pop("SecuritiesAccount")
    data_dict["CD Account"] = data_dict.pop("CDAccount")
    df = pd.DataFrame([data_dict])

    # Predict
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    # SHAP Values
    X_transformed = model.named_steps['preprocessing'].transform(df)
    shap_values = explainer(X_transformed)
    shap_values[0].feature_names = df.columns.tolist()

    # Get SHAP values for top 5 features
    top_features = dict(
        zip(df.columns, shap_values[0].values)
    )
    sorted_top = dict(sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

    # Optional: SHAP force plot image as base64
    fig = shap.plots.waterfall(shap_values[0], show=False)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "prediction": int(prediction),
        "probability": float(round(prob, 4)),
        "top_shap_features": {k: float(v) for k, v in sorted_top.items()},
        "shap_plot_base64": img_base64
    }

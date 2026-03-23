from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import uvicorn

# -------------------------------
# 1. Load Data
# -------------------------------

df = pd.read_csv("insurance.csv")

X = df.drop("charges", axis=1)
y = df["charges"]

# Preprocessing columns
standard_scalers_columns = ['age', 'bmi']
binary_encode_columns = ['sex', 'smoker']
onehot_columns = ['region']

# -------------------------------
# 2. Preprocessing + Model Pipeline
# -------------------------------

preprocessing = ColumnTransformer(
    transformers=[
        ("std", StandardScaler(), standard_scalers_columns),
        ("onehot", OneHotEncoder(drop="first"), onehot_columns),
        ("binary", OneHotEncoder(drop="first"), binary_encode_columns)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ("preprocessor", preprocessing),
    ("regressor", LinearRegression())
])

# -------------------------------
# 3. Train Model
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model is trained successfully!")

# -------------------------------
# 4. FastAPI Setup
# -------------------------------

app = FastAPI(title="Insurance Charges Prediction API")

# Pydantic model for input request body
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# -------------------------------
# 5. Prediction Endpoint
# -------------------------------

@app.post("/predict")
def predict_charges(data: InsuranceInput):

    # Convert input to dataframe
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_df)[0]

    return {
        "predicted_charges": float(prediction)
    }


# Run app (for local testing)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

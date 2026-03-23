from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import uvicorn

# ---------------------------------
# 1. Load and Train Model
# ---------------------------------

df = pd.read_csv("insurance.csv")

X = df.drop("charges", axis=1)
y = df["charges"]

standard_scalers_columns = ["age", "bmi"]
binary_encode_columns = ["sex", "smoker"]
onehot_columns = ["region"]

# Preprocessor
preprocessing = ColumnTransformer(
    transformers=[
        ("std", StandardScaler(), standard_scalers_columns),
        ("onehot", OneHotEncoder(drop="first"), onehot_columns),
        ("binary", OneHotEncoder(drop="first"), binary_encode_columns)
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessing),
    ("regressor", LinearRegression())
])

# Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

print("✅ Model trained successfully")


# ---------------------------------
# 2. FastAPI App
# ---------------------------------

app = FastAPI(title="Insurance Charges Prediction API")


# Allowed categorical values
ALLOWED_SEX = {"male", "female"}
ALLOWED_SMOKER = {"yes", "no"}
ALLOWED_REGION = {"southeast", "southwest", "northeast", "northwest"}


# Pydantic model + validation
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

    # Validators
    @validator("sex")
    def validate_sex(cls, v):
        if v.lower() not in ALLOWED_SEX:
            raise ValueError(f"sex must be one of {ALLOWED_SEX}")
        return v.lower()

    @validator("smoker")
    def validate_smoker(cls, v):
        if v.lower() not in ALLOWED_SMOKER:
            raise ValueError(f"smoker must be one of {ALLOWED_SMOKER}")
        return v.lower()

    @validator("region")
    def validate_region(cls, v):
        if v.lower() not in ALLOWED_REGION:
            raise ValueError(f"region must be one of {ALLOWED_REGION}")
        return v.lower()

    @validator("age")
    def validate_age(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("age must be between 1 and 100")
        return v

    @validator("bmi")
    def validate_bmi(cls, v):
        if v <= 0:
            raise ValueError("bmi must be a positive number")
        return v

    @validator("children")
    def validate_children(cls, v):
        if v < 0 or v > 10:
            raise ValueError("children must be between 0 and 10")
        return v


# ---------------------------------
# 3. Prediction Route with Error Handling
# ---------------------------------

@app.post("/predict")
def predict(data: InsuranceInput):
    try:
        input_df = pd.DataFrame([data.dict()])

        prediction = model.predict(input_df)[0]

        return {
            "predicted_charges": float(prediction)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


# ---------------------------------
# For Local Testing
# ---------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

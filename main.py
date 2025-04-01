from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import requests
from io import BytesIO

app = FastAPI(
    title="Titanic Survival Predictor",
    description="API for predicting survival of Titanic passengers",
    version="1.0.0"
)

# Load model 
MODEL_URL = "https://github.com/DwaipayanDutta/Titanic_App/raw/main/Model/titanic_model.pkl"
DATA_URL = "https://github.com/DwaipayanDutta/Titanic_App/raw/main/Data/titanic.csv"

# Download and load model
model_response = requests.get(MODEL_URL)
model = joblib.load(BytesIO(model_response.content))

# Load and preprocess data
LOOKUP_DF = pd.read_csv(DATA_URL)
LOOKUP_DF = LOOKUP_DF[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Create preprocessing pipeline
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Pclass', 'Sex', 'Embarked']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit preprocessor on entire dataset
preprocessor.fit(LOOKUP_DF.drop('PassengerId', axis=1))

class PassengerRequest(BaseModel):
    PassengerId: int 

@app.post("/predict", summary="Predict survival status")
async def predict_survival(passenger: PassengerRequest):
    try:
        passenger_id = str(passenger.PassengerId)
        passenger_data = LOOKUP_DF[LOOKUP_DF['PassengerId'].astype(str) == passenger_id]
        
        if passenger_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Passenger ID {passenger.PassengerId} not found in records"
            )
            
        # Preprocess the data
        features = passenger_data.drop('PassengerId', axis=1)
        processed_features = preprocessor.transform(features)
        
        # Make prediction
        prediction = model.predict(processed_features)
        probability = model.predict_proba(processed_features)[0].max()
        
        return {
            "passenger_id": passenger.PassengerId,
            "survival_status": "Survived" if prediction[0] == 1 else "Did not survive",
            "confidence": round(float(probability), 3),
            "features": features.iloc[0].to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

🚢 Titanic Survival Predictor (FastAPI + Swagger Demo)
This project is a FastAPI application that predicts whether a passenger survived the Titanic disaster. It uses a pre-trained machine learning model and provides a simple API interface documented via Swagger.

📦 Requirements
Before running the app, install the required Python packages:

bash
Copy
Edit
pip install fastapi uvicorn pandas scikit-learn joblib
Dependencies
FastAPI – For building the REST API.

Uvicorn – ASGI server to run the FastAPI app.

Pandas – For data manipulation and lookup.

Scikit-learn – For loading and running the ML model.

Joblib – For model serialization/deserialization.

Pickle – (Used internally) to handle object serialization.

📁 Project Structure
bash
Copy
Edit
├── app.py                     # Main FastAPI app
├── model/
│   └── titanic_model.pkl      # Pre-trained ML model
└── Data/
    └── titanic.csv            # Titanic passenger dataset
⚙️ How It Works
Model Loading
The app loads a pre-trained model from model/titanic_model.pkl using joblib.

Data Lookup
Passenger data is fetched from a dataset hosted on GitHub. The dataset contains key features like age, sex, class, fare, etc.

Prediction
A POST request to the /predict endpoint with a PassengerId retrieves the passenger's data and predicts whether they survived.

🔌 API Endpoint
POST /predict
Predicts survival based on the given PassengerId.

📥 Request Example
json
Copy
Edit
{
  "PassengerId": "1"
}
📤 Successful Response Example
json
Copy
Edit
{
  "passenger_id": "1",
  "survival_status": "Survived",
  "features": {
    "Pclass": 1,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 71.2833,
    "Embarked": "C"
  }
}
❌ Error Responses
Passenger not found:

json
Copy
Edit
{
  "detail": "Passenger ID 9999 not found in records"
}
Prediction error:

json
Copy
Edit
{
  "detail": "Prediction error: [Not Found]"
}
▶️ Running the Application
Start the FastAPI server locally using uvicorn:

bash
Copy
Edit
uvicorn app:app --reload
The app will be available at: http://127.0.0.1:8001

🧪 Example cURL Request
bash
Copy
Edit
curl -X 'POST' 'http://127.0.0.1:8001/predict' \
-H 'Content-Type: application/json' \
-d '{ "PassengerId": "1" }'
⚠️ Notes
The model was trained on historical Titanic passenger data including fields like age, class, sex, fare, and embarkation point.

The dataset is pulled from a remote CSV file:
https://raw.githubusercontent.com/DwaipayanDutta/Titanic_App/refs/heads/main/Data/titanic.csv

Ensure this link is accessible when running the app.

🚀 Optional: Airflow Integration
To orchestrate this prediction workflow using Apache Airflow:

1. Install Airflow
bash
Copy
Edit
pip install apache-airflow
airflow db init
airflow users create \
  --username admin --firstname Admin --lastname User \
  --role Admin --email admin@example.com
airflow scheduler
airflow webserver --port 8080
2. Prepare Resources
Confirm the model (titanic_model.pkl) is in the correct model/ directory.

Ensure the dataset URL is available.

3. Monitor Workflow
Use the Airflow UI (http://localhost:8080) to:

Monitor job runs

Trigger predictions manually or on a schedule

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
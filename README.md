# Titanic Survival Predictor

This FastAPI app provides a service to predict whether a passenger survived the Titanic disaster based on certain features like age, gender, class, etc. It utilizes a pre-trained machine learning model to make these predictions.

## Requirements

Before running this application, you need to install the following dependencies:

- **FastAPI**: The web framework to build the API.
- **Uvicorn**: ASGI server for running FastAPI apps.
- **Pandas**: Data manipulation and analysis library.
- **Scikit-learn**: For loading and using the pre-trained machine learning model.
- **Joblib**: To handle serialization and deserialization of the model.
- **Pickle**: Python’s built-in object serialization tool (used internally for model loading).

You can install the required dependencies using pip:

`pip install fastapi uvicorn pandas scikit-learn joblib`

## Project Structure

- **app.py**: The FastAPI application containing the prediction logic.
- **model/titanic_model.pkl**: The pre-trained model file.
- **Data/titanic.csv**: The dataset used for prediction lookups.

## How It Works

1. **Model Loading**: The model is loaded from a `.pkl` file using `joblib`.
2. **Dataset**: The dataset is fetched from a remote CSV file hosted on GitHub. This dataset is used to lookup passenger features for making predictions.
3. **Prediction**: The API endpoint accepts a `POST` request with a passenger's ID, retrieves the corresponding passenger data, and uses the model to predict if the passenger survived or not.

## API Endpoints

### POST `/predict`

This endpoint accepts a `POST` request with a JSON body containing a `PassengerId`, and returns whether the passenger survived the Titanic disaster or not, along with relevant features.

#### Request Body Example

{
  "PassengerId": "1"
}

#### Response Example

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

#### Error Response Example

If the `PassengerId` is not found in the dataset:

{
  "detail": "Passenger ID 9999 not found in records"
}

If there is an error during the prediction:

{
  "detail": "Prediction error: [Not Found]"
}

## Run the Application

To run the application locally, use `uvicorn`:

`uvicorn app:app --reload`

This will start the FastAPI server on `http://127.0.0.1:8001`, and you can send requests to it.

## Example cURL Command

You can use `curl` to test the prediction endpoint:

`curl -X 'POST' 'http://127.0.0.1:8001/predict' -H 'Content-Type: application/json' -d '{ "PassengerId": "1" }'`

## Notes

- The model used here is a pre-trained Titanic survival prediction model (`titanic_model.pkl`), which was trained on passenger data like age, sex, class, and other features.
- Ensure that the dataset URL used (`https://raw.githubusercontent.com/DwaipayanDutta/Titanic_App/refs/heads/main/Data/titanic.csv`) is accessible when running the app.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

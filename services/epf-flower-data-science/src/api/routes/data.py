import kaggle
from fastapi import APIRouter, HTTPException, Request
# from src.schemas.message import MessageResponse
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import os
# from google.cloud import firestore
import firestore

router = APIRouter()



@router.get("/download-dataset")
async def download_dataset():
    # The authenticate method no longer requires 'username' and 'key' parameters
    kaggle.api.authenticate()
    
    dataset_name = "uciml/iris"
    save_dir = "services\epf-flower-data-science\src\data"
    kaggle.api.dataset_download_files(dataset_name, path=save_dir, unzip=True)

    return {"message": "Dataset downloaded and saved successfully."}

@router.get("/load-data")
def load_data():
    file_path = "services\epf-flower-data-science\src\data\Iris.csv"
    df = pd.read_csv(file_path)
    return df.to_json(orient="records")

@router.get("/preprocess")
def preprocess_data():
    data = load_data()
    data = data.replace("Iris-", "")
    data = data.replace("Cm","(Cm)")
    return data

@router.get("/split")
def train_test_split_func():
    data  = pd.read_json(preprocess_data())
    train, test = train_test_split(data, test_size=0.2)
    return train.to_json(orient="records"), test.to_json(orient="records")

@router.get("/init-params")
def init_params():
    model = SVC()
    model_params = model.get_params()
    file_path = 'services\epf-flower-data-science\src\config\model_parameters.json'

    # Save the model parameters to a JSON file
    with open(file_path, 'w') as file:
        json.dump(model_params, file, indent=2)

@router.get("/train-model")
def train_model():
    try:
        train, _ = train_test_split_func()
        train_data = pd.read_json(train)
        
        # Extract features and labels
        features = train_data.drop(columns=["Species"])
        labels = train_data["Species"]

        # Convert labels to numeric format using LabelEncoder
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)

        # Load the model parameters from the JSON file
        model_params_file = 'services\epf-flower-data-science\src\config\model_parameters.json'
        with open(model_params_file, 'r') as file:
            model_params = json.load(file)

        # Initialize the SVC classifier with loaded parameters
        model = SVC(**model_params)

        # Train the model
        model.fit(features, labels_encoded)
        if not os.path.exists('services/epf-flower-data-science/src/models'):
            os.makedirs('services/epf-flower-data-science/src/models')
        # Save the trained model to the models folder
        model_filename = 'services/epf-flower-data-science/src/models/trained_model.joblib'
        joblib.dump(model, model_filename)

        return {"message": "Model trained and saved successfully"}

    except Exception as e:
        return {"error": str(e)}
    
@router.get("/predict")
def predict():
    model_save_path = 'services/epf-flower-data-science/src/models/trained_model.joblib'
    try:
        model = joblib.load(model_save_path)
    except FileNotFoundError:
        return {"error": "Trained model not found."}
    
    train, test = train_test_split_func()
    test_df = pd.read_json(test)
    
    X_test = test_df.drop(columns=["Species"])
    y_pred = pd.DataFrame(model.predict(X_test))

    return y_pred.to_json(orient="records")
    
@router.get("/get-firestore-data")
def get_firestore_data():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "services\epf-flower-data-science\src\config\credentials.json"
    # Initialize Firestore
    db = firestore.Client()

    # Reference the collection and document
    collection_ref = db.collection("parameters")
    document_ref = collection_ref.document("parameters")

    # Get data from the document
    doc_data = document_ref.get().to_dict()

    # Check if data was retrieved
    if doc_data:
        return doc_data
    else:
        return None  
    
@router.get("/update-firestore-data")
def update_firestore_data(parameter_name, parameter_value):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "services\epf-flower-data-science\src\config\credentials.json"
    # Initialize Firestore
    db = firestore.Client()

    # Reference the collection and document
    collection_ref = db.collection("parameters")
    document_ref = collection_ref.document("parameters")

    # Get existing data from the document
    doc_data = document_ref.get().to_dict() 

    # Add or update the parameter
    doc_data[parameter_name] = parameter_value
    document_ref.set(doc_data)

    return {"Firestore parameter edited with success"}
    
    
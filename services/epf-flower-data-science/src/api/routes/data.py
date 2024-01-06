from src.services.data import download_dataset, load_data, preprocess_data, train_test_split_func, init_params, train_model, predict, get_firestore_data, update_firestore_data
from fastapi import APIRouter, HTTPException, Request


router = APIRouter()



@router.get("/download-dataset")
async def download_dataset_route():

    return download_dataset()

@router.get("/load-data")
def load_data_route():
    return load_data()

@router.get("/preprocess")
def preprocess_data_route():
    return preprocess_data()

@router.get("/split")
def train_test_split_func_route():
    return train_test_split_func()

@router.get("/init-params")
def init_params_route():
    return init_params()

@router.get("/train-model")
def train_model_route():
    return train_model()
    
@router.get("/predict")
def predict_route():
    return predict()
    
@router.get("/get-firestore-data")
def get_firestore_data_route():
    return get_firestore_data()
    
@router.get("/update-firestore-data")
def update_firestore_data_route():
    return update_firestore_data('test_param',400)
    
    
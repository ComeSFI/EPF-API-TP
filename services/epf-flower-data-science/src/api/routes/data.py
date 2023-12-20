import kaggle
from fastapi import APIRouter
# from src.schemas.message import MessageResponse
import pandas as pd
from sklearn.model_selection import train_test_split

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


    
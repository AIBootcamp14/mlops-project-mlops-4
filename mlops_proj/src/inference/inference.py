import os
import sys
import glob
import pickle

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils.utils import model_dir, calculate_hash, read_hash
from src.model.movie_predictor import movie_predictor
from src.dataset.watch_log import watch_log_dataset, get_datasets
from src.dataset.data_loader import data_loader
from src.evaluate.evaluate import evaluate
from src.postprocess.postprocess import write_db


def model_validation(model_path):
    original_hash = read_hash(model_path)
    current_hash = calculate_hash(model_path)

    if original_hash == current_hash:
        print("validation success")
        return True
    return False

def load_checkpoint():
    target_dir = model_dir(movie_predictor.name)
    models_path = os.path.join(target_dir, "*.pkl")
    latest_model = glob.glob(models_path)[-1]

    if not model_validation(latest_model):
        raise FileExistsError("Not found or invalid model file")
    
    with open(latest_model, "rb") as f:
        checkpoint = pickle.load(f)
    return checkpoint

def init_model(checkpoint):
    model = movie_predictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = checkpoint.get("scaler", None)
    label_encoder = checkpoint.get("label_encoder", None)
    return model, scaler, label_encoder

def make_inference_df(data):
    columns = "user_id content_id watch_seconds rating popularity".split()
    return pd.DataFrame(
        data = [data],
        columns = columns
    )

def inference(model, scaler, label_encoder, data: np.array, batch_size = 1):
    if data.size > 0: # online inference
        df = make_inference_df(data)
        dataset = watch_log_dataset(df, scaler = scaler, label_encoder = label_encoder)
    else: # offline(batch) inference
        _, _, dataset = get_datasets(scaler = scaler, label_encoder = label_encoder)
    dataloader = data_loader(
		    dataset.features, 
            dataset.labels, 
            batch_size=batch_size, 
            shuffle=False
		)
    loss, predictions = evaluate(model, dataloader)
    print(loss, predictions)
    return [dataset.decode_content_id(idx) for idx in predictions]

def recommend_to_df(recommend):
    return pd.DataFrame(
        data=recommend, # [12345, 2345435, 123923, 120390]
        columns="recommend_content_id".split()
    )


if __name__ == '__main__':
    load_dotenv()
    checkpoint = load_checkpoint()
    model, scaler, label_encoder = init_model(checkpoint)
    #data = np.array([1, 1209290, 4508, 7.577, 1204.764])
    data = np.array([])
    batch_size = 32
    recommend = inference(model, scaler, label_encoder, data, batch_size = batch_size)
    print(recommend)
    recommend_df = recommend_to_df(recommend)
    write_db(recommend_df, "mlops", "recommend")


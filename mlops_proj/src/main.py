import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import wandb
import fire
import numpy as np
from icecream import ic
from dotenv import load_dotenv

from src.dataset.watch_log import get_datasets
from src.dataset.data_loader import data_loader
from src.model.movie_predictor import movie_predictor, model_save
from src.utils.utils import init_seed, auto_increment_run_suffix, parse_date
from src.utils.enums import model_types
from src.train.train import train
from src.evaluate.evaluate import evaluate
from src.inference.inference import load_checkpoint, init_model, inference, recommend_to_df
from src.postprocess.postprocess import write_db


load_dotenv()
init_seed()

def get_runs(project_name):
    return wandb.Api().runs(path=project_name, order = "-created_at")

def get_latest_run(project_name):
    runs = get_runs(project_name)
    if not runs:
        return f"{project_name}-000"
    return runs[0].name

def run_train(model_name, num_epochs = 10, batch_size = 64, hidden_dim = 64):
	# Validate model type
    model_types.validation(model_name)

	# Connect to WandB
    api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=api_key)

    project_name = model_name.lower().replace('_', '-')

    run_name = get_latest_run(project_name)
    next_run_name = auto_increment_run_suffix(run_name)

    wandb.init(
        project = project_name, 
        id = next_run_name, 
        name = next_run_name, 
        notes = "content-based movie recommend model", 
        tags = ["content-based", "movie", "recommend"], 
        config = locals(),
        reinit = True, 
	)

    # Create dataset and data loader
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = data_loader(train_dataset.features, train_dataset.labels, batch_size = batch_size, shuffle = True)
    val_loader = data_loader(val_dataset.features, val_dataset.labels, batch_size = batch_size, shuffle = False)
    test_loader = data_loader(test_dataset.features, test_dataset.labels, batch_size = batch_size, shuffle = False)

    # Init model
    model_params = {
        "input_dim": train_dataset.features_dim, 
        "num_classes": train_dataset.num_classes, 
        "hidden_dim": hidden_dim, 
    }
    # model = movie_predictor(**model_params)
    model_class = model_types[model_name.upper()].value # MOVIE_PREDICTOR = movie_predictor
    model = model_class(**model_params)

    # Train loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader)
        val_loss, _ = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val-Train Loss : {val_loss-train_loss:.4f}")
        wandb.log({"Loss/Train": train_loss})
        wandb.log({"Loss/Valid": val_loss})

    # Test
    test_loss, predictions = evaluate(model, test_loader)
    #print(f"Test Loss : {test_loss:.4f}")
    ic(test_loss)
    #ic(predictios)
    #print([train_dataset.decode_content_id(idx) for idx in predictions])

    # Save model
    model_save(
        model = model,
        model_params = model_params,
        epoch = num_epochs,
        loss = train_loss,
        scaler = train_dataset.scaler,
        label_encoder = train_dataset.label_encoder,
    )

    # WandB 세션 종료 및 로그아웃
    wandb.finish()

def run_inference(data = None, batch_size = 64):
    checkpoint = load_checkpoint()
    model, scaler, label_encoder = init_model(checkpoint)

    if data is None:
        data = []

    data = np.array(data)

    recommend = inference(model, scaler, label_encoder, data, batch_size)
    print(recommend)

    write_db(recommend_to_df(recommend), "mlops", "recommend")

def run_preprocessing(date):
    parsed_date = parse_date(date)
    print(f"Run date : {parsed_date.year}. {parsed_date.month}. {parsed_date.day}")
    print("Run some preprocessing...")
    print("Done!")


if __name__ == '__main__':
    # Command example) python src/main.py train --model_name movie_predictor --num_epochs 20 --hidden_dim 32
    fire.Fire({
        "preprocessing": run_preprocessing, 
        "train": run_train,
        "inference": run_inference,
    })


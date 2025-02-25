# ml/model_helper.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import os
import io
import logging
import warnings
import tensorflow as tf
from datetime import datetime
import json
import pickle
import base64
import time
import threading
from bson import ObjectId
import subprocess
import sys

# Import ML modules and database utilities
import ml.AE_tools as AE_tools
from ml.AE import AutoEncoder
from db_utils.db import (
    db,
    create_document,
    read_documents,
    read_document,
    update_document,
    delete_document,
)


# ---------------------------
# Utility Functions for MongoDB Model Operations
# ---------------------------
def save_model_to_db(
    model_name,
    description,
    source_collection,
    hyperparameters,
    model_obj,
    scaler_obj,
    error_threshold,
    history=None,
):
    """
    Save a trained model to MongoDB with all relevant metadata.
    """
    # Serialize the model and scaler
    serialized_model = serialize_model(model_obj)
    serialized_scaler = serialize_scaler(scaler_obj)

    # Create the model document
    model_doc = {
        "name": model_name,
        "created_at": datetime.now(),
        "description": description,
        "source_collection": source_collection,
        "hyperparameters": hyperparameters,
        "error_threshold": error_threshold,
        "model": serialized_model,
        "scaler": serialized_scaler,
        "training_status": "completed",
        "training_progress": 1.0,
        "training_time": None,
    }

    # Add history metrics if available
    if history is not None:
        model_doc["metrics"] = {
            "loss": history.get("loss", []),
            "val_loss": history.get("val_loss", []),
        }

    # Save to MongoDB
    model_id = create_document(model_doc, "autoencoder_models")
    return model_id


def load_model_from_db(model_id):
    """
    Load a model from MongoDB by its ID.
    """
    model_doc = read_document({"_id": ObjectId(model_id)}, "autoencoder_models")
    if not model_doc:
        raise ValueError(f"Model with ID {model_id} not found")

    # Deserialize the model and scaler
    model = load_model_from_serialized(model_doc["model"])
    scaler = load_scaler_from_serialized(model_doc["scaler"])

    return model, scaler, model_doc


def get_all_models():
    """
    Get all models from the MongoDB collection.
    """
    return read_documents({}, "autoencoder_models")


def update_training_progress(model_id, progress, status="in_progress"):
    """
    Update the training progress of a model in MongoDB.
    """
    update_document(
        {"_id": ObjectId(model_id)},
        {"training_progress": progress, "training_status": status},
        "autoencoder_models",
    )


def create_training_job(model_name, description, source_collection, hyperparameters):
    """
    Create a new training job entry in MongoDB.
    """
    job_doc = {
        "name": model_name,
        "created_at": datetime.now(),
        "description": description,
        "source_collection": source_collection,
        "hyperparameters": hyperparameters,
        "training_status": "initialized",
        "training_progress": 0.0,
        "training_time": None,
    }
    job_id = create_document(job_doc, "autoencoder_models")
    return str(job_id)


def delete_model(model_id):
    """
    Delete a model from MongoDB by its ID.
    """
    return delete_document({"_id": ObjectId(model_id)}, "autoencoder_models")


# ---------------------------
# Utility Function: Load Data from DB
# ---------------------------
def load_data_from_db(collection_name: str):
    """
    Load all documents from the given MongoDB collection and return them as a DataFrame.
    """
    docs = read_documents(query={}, collection_name=collection_name)
    if not docs:
        st.error("No documents found in the selected collection!")
        return None
    return pd.DataFrame(docs)


# ---------------------------
# Model and Scaler Serialization Functions
# ---------------------------
def serialize_model(model):
    model_json = model.to_json()
    model_weights = [w.tolist() for w in model.get_weights()]
    return {"architecture": model_json, "weights": model_weights}


def load_model_from_serialized(serialized_model):
    model = tf.keras.models.model_from_json(serialized_model["architecture"])
    weights = [np.array(w) for w in serialized_model["weights"]]
    model.set_weights(weights)
    try:
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=["acc"])
    except Exception as e:
        logging.info("Compilation error while loading model: %s", e)
    return model


def serialize_scaler(scaler):
    return base64.b64encode(pickle.dumps(scaler)).decode("utf-8")


def load_scaler_from_serialized(scaler_str):
    return pickle.loads(base64.b64decode(scaler_str.encode("utf-8")))


# ---------------------------
# Async Training Functions
# ---------------------------
def train_model_async(model_id, df, training_devices, hyperparams):
    """
    Launch model training as a separate process instead of a thread
    """
    # Save temporary data for the training process
    temp_data_path = f"temp_data_{model_id}.pkl"
    temp_config_path = f"temp_config_{model_id}.json"

    # Save training data and configuration
    with open(temp_data_path, "wb") as f:
        pickle.dump(
            {"df": df, "training_devices": training_devices, "hyperparams": hyperparams},
            f,
        )

    # Create a configuration file
    with open(temp_config_path, "w") as f:
        json.dump({"model_id": model_id, "data_path": temp_data_path}, f)

    # Launch the training script as a separate process
    subprocess.Popen([sys.executable, "train_model.py", temp_config_path])

    # Update status to indicate training has been queued
    update_document(
        {"_id": ObjectId(model_id)},
        {
            "training_status": "in_progress",
            "training_progress": 0.0,
            "start_time": datetime.now(),
        },
        "autoencoder_models",
    )

    return model_id

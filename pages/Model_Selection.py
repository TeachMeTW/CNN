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
import ml.AE_tools as AE_tools
import tempfile

# Import helper functions
from ml.model_helper import (
    db,
    create_document,
    read_documents,
    read_document,
    update_document,
    delete_document,
    load_data_from_db,
    serialize_model,
    load_model_from_serialized,
    serialize_scaler,
    train_model_async,
    save_model_to_db,
    load_model_from_db,
    get_all_models,
    update_training_progress,
    create_training_job,
    delete_model,
)

# ---------------------------
# Configure Logging & Warnings
# ---------------------------
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
warnings.filterwarnings("ignore")

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="IoT AutoEncoder Dashboard", layout="wide")
st.title("Model Selection: IoT AutoEncoder Dashboard")

# ---------------------------
# Session State Variables
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "df_copy" not in st.session_state:
    st.session_state.df_copy = None
if "FEATURES" not in st.session_state:
    st.session_state.FEATURES = []
if "CONTEXT" not in st.session_state:
    st.session_state.CONTEXT = []
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "train_df" not in st.session_state:
    st.session_state.train_df = None
if "ERROR_THRESHOLD" not in st.session_state:
    st.session_state.ERROR_THRESHOLD = None
if "history" not in st.session_state:
    st.session_state.history = None
if "hyperparameters" not in st.session_state:
    st.session_state.hyperparameters = {}
if "source_collection" not in st.session_state:
    st.session_state.source_collection = None
if "training_job" not in st.session_state:
    st.session_state.training_job = None
if "active_model_id" not in st.session_state:
    st.session_state.active_model_id = None
if "cancel_training" not in st.session_state:
    st.session_state.cancel_training = False
if "cancel_training" not in st.session_state:
    st.session_state.cancel_training = False

# ---------------------------
# Constants for Training
# ---------------------------
SCALER = "Robust"
MACHINE_TYPE = "mac"
SAMPLE_PERC = 0.5

# ---------------------------
# Create Tabs (5 steps)
# ---------------------------
tabs = st.tabs(
    [
        "Models",
        "Configuration",
        "Data Exploration",
        "Results & Visualization",
        "Anomalies",
    ]
)

# ====================================================
# TAB 0: MODELS (New tab for model management)
# ====================================================
with tabs[0]:
    st.subheader("Model Management")
    st.info(
        "Manage your AutoEncoder models: create, view, train, and delete models."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("### Models")
        all_models = get_all_models()

        if st.button("Refresh Model List"):
            st.rerun()

        if not all_models:
            st.info("No models found. Create a new model to get started.")
        else:
            st.write("Select a model to view details:")
            for model in all_models:
                status_indicator = {
                    "completed": "✅",
                    "in_progress": "🔄",
                    "cancelled": "❌",
                    "failed": "⚠️",
                    "initialized": "🔍",
                }.get(model.get("training_status", ""), "")

                progress = model.get("training_progress", 0)
                progress_str = (
                    f" - {int(progress * 100)}%"
                    if model.get("training_status") == "in_progress"
                    else ""
                )
                if st.button(
                    f"{status_indicator} {model['name']}{progress_str}",
                    key=f"model_{model['_id']}",
                ):
                    st.session_state.active_model_id = str(model["_id"])

        st.write("### Create New Model")
        if st.button("➕ Create New Model", key="new_model_btn"):
            st.session_state.active_model_id = "new"

    with col2:
        # ----------------------------------------------------------------
        # Creating a new model
        # ----------------------------------------------------------------
        if st.session_state.active_model_id == "new":
            st.write("### New Model Configuration")

            model_name = st.text_input("Model Name", value="My AutoEncoder Model")
            model_desc = st.text_area(
                "Description", value="Anomaly detection model for IoT sensors"
            )
            all_collections = db.list_collection_names()
            data_collections = [
                col
                for col in all_collections
                if col not in ("experiment_logs", "autoencoder_models")
            ]
            if not data_collections:
                st.error("No data collections found in the database!")
                source_collection = None
            else:
                source_collection = st.selectbox(
                    "Data Source:",
                    data_collections,
                    help="Select the collection containing the training data.",
                )

            # Select training devices
            training_devices = []
            if source_collection:
                with st.spinner("Loading data to get device list..."):
                    temp_df = load_data_from_db(source_collection)
                    if temp_df is not None and "device" in temp_df.columns:
                        all_devices = temp_df["device"].unique().tolist()
                        training_devices = st.multiselect(
                            "Training Devices:",
                            options=all_devices,
                            default=all_devices,
                            help="Select devices to use for training.",
                        )

            # Preset / Custom hyperparameters
            hyperparameter_mode = st.radio(
                "Hyperparameter Profile:",
                ["Default", "Fast", "Expensive", "Custom"],
                help="Choose a preset profile or custom settings.",
            )
            hyperparameter_profiles = {
                "Default": {
                    "latent_dim": 5,
                    "dropout": 0.1,
                    "reduction_modulo": 2,
                    "epochs": 300,
                    "batch_size": 128,
                    "learning_rate": 0.005,
                    "error_threshold_percentile": 99.95,
                    "description": "Balanced settings for general anomaly detection",
                },
                "Fast": {
                    "latent_dim": 3,
                    "dropout": 0.05,
                    "reduction_modulo": 3,
                    "epochs": 200,
                    "batch_size": 256,
                    "learning_rate": 0.01,
                    "error_threshold_percentile": 99.90,
                    "description": "Optimized for quick training and deployment",
                },
                "Expensive": {
                    "latent_dim": 8,
                    "dropout": 0.15,
                    "reduction_modulo": 1,
                    "epochs": 1500,
                    "batch_size": 64,
                    "learning_rate": 0.0005,
                    "error_threshold_percentile": 99.99,
                    "description": "High-accuracy settings with longer training time",
                },
            }

            if hyperparameter_mode == "Custom":
                st.info("Adjust the hyperparameters for the AutoEncoder model below.")
                with st.expander("Custom Hyperparameters", expanded=True):
                    colA, colB = st.columns(2)
                    with colA:
                        latent_dim = st.slider(
                            "Latent Dimension", min_value=2, max_value=20, value=5
                        )
                        dropout = st.slider(
                            "Dropout (%)",
                            min_value=0.01,
                            max_value=0.9,
                            value=0.1,
                            step=0.01,
                        )
                        reduction_modulo = st.slider(
                            "Reduction Modulo", min_value=1, max_value=10, value=2
                        )
                    with colB:
                        epochs = st.selectbox(
                            "Epochs", options=[200, 500, 1000, 1500, 2000], index=1
                        )
                        batch_size = st.selectbox(
                            "Batch Size", options=[64, 128, 256, 512, 1000], index=1
                        )
                        learning_rate = st.selectbox(
                            "Learning Rate",
                            options=[0.1, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001],
                            index=4,
                        )

                    error_threshold = st.slider(
                        "Error Threshold (Percentile)",
                        min_value=90.0,
                        max_value=99.999,
                        value=99.95,
                        step=0.001,
                    )

                hyperparams = {
                    "latent_dim": latent_dim,
                    "dropout": dropout,
                    "reduction_modulo": reduction_modulo,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "error_threshold_percentile": error_threshold,
                    "profile": "Custom",
                }
            else:
                profile = hyperparameter_profiles[hyperparameter_mode]
                st.info(f"**{hyperparameter_mode} Profile**: {profile['description']}")
                st.write(
                    f"• Epochs: {profile['epochs']} | Latent Dim: {profile['latent_dim']} "
                    f"| Error Threshold: {profile['error_threshold_percentile']}%"
                )
                with st.expander("Show all parameters"):
                    colA, colB = st.columns(2)
                    with colA:
                        st.write("**Model Parameters:**")
                        st.write(f"• Latent Dimension: {profile['latent_dim']}")
                        st.write(f"• Dropout: {profile['dropout']}")
                        st.write(f"• Reduction Modulo: {profile['reduction_modulo']}")
                        st.write(
                            f"• Error Threshold: {profile['error_threshold_percentile']} percentile"
                        )
                    with colB:
                        st.write("**Training Parameters:**")
                        st.write(f"• Epochs: {profile['epochs']}")
                        st.write(f"• Batch Size: {profile['batch_size']}")
                        st.write(f"• Learning Rate: {profile['learning_rate']}")

                hyperparams = {
                    "latent_dim": profile["latent_dim"],
                    "dropout": profile["dropout"],
                    "reduction_modulo": profile["reduction_modulo"],
                    "epochs": profile["epochs"],
                    "batch_size": profile["batch_size"],
                    "learning_rate": profile["learning_rate"],
                    "error_threshold_percentile": profile["error_threshold_percentile"],
                    "profile": hyperparameter_mode,
                }

            # Create model & train
            if st.button("Create and Start Training"):
                job_id = create_training_job(
                    model_name=model_name,
                    description=model_desc,
                    source_collection=source_collection,
                    hyperparameters=hyperparams,
                    training_devices=training_devices,
                )
                train_model_async(job_id)
                st.success(f"Training started for {model_name}")
                st.rerun()

        # ----------------------------------------------------------------
        # An existing model is selected
        # ----------------------------------------------------------------
        elif st.session_state.active_model_id:
            try:
                model_doc = read_document(
                    {"_id": ObjectId(st.session_state.active_model_id)},
                    "autoencoder_models",
                )
                if model_doc:
                    st.write(f"### {model_doc['name']}")
                    st.write(
                        f"**Description:** {model_doc.get('description', 'No description')}"
                    )
                    st.write(f"**Created:** {model_doc.get('created_at', 'Unknown')}")
                    st.write(
                        f"**Source Data:** {model_doc.get('source_collection', 'Unknown')}"
                    )

                    status = model_doc.get("training_status", "Unknown")
                    status_labels = {
                        "initialized": "🔍 Initialized",
                        "in_progress": "🔄 Training in Progress",
                        "completed": "✅ Training Complete",
                        "cancelled": "❌ Training Cancelled",
                        "failed": "⚠️ Training Failed",
                    }
                    st.write(f"**Status:** {status_labels.get(status, status)}")

                    # Delete model
                    if st.button("🗑️ Delete Model", key="delete_model"):
                        if (
                            st.session_state.active_model_id
                            == st.session_state.get("training_job")
                        ):
                            st.error("Cannot delete a model that is currently training!")
                        else:
                            delete_model(st.session_state.active_model_id)
                            st.session_state.active_model_id = None
                            st.success("Model deleted successfully!")
                            st.rerun()

                    # Show progress if training
                    if status == "in_progress":
                        progress = model_doc.get("training_progress", 0)
                        st.progress(progress)
                        st.write(f"Progress: {int(progress * 100)}%")

                        # Cancel training
                        if st.button("❌ Cancel Training", key="cancel_training"):
                            update_document(
                                {"_id": ObjectId(st.session_state.active_model_id)},
                                {"cancel_requested": True},
                                "autoencoder_models",
                            )
                            st.warning("Cancellation requested. The process will stop soon.")

                    # Failed message
                    if status == "failed" and "error_message" in model_doc:
                        st.error(f"Error: {model_doc['error_message']}")

                    # Show hyperparams
                    if "hyperparameters" in model_doc:
                        with st.expander("Hyperparameters", expanded=False):
                            hyperparams = model_doc["hyperparameters"]
                            colA, colB = st.columns(2)
                            with colA:
                                st.write("**Model Parameters:**")
                                st.write(f"• Latent Dimension: {hyperparams.get('latent_dim','N/A')}")
                                st.write(f"• Dropout: {hyperparams.get('dropout','N/A')}")
                                st.write(f"• Reduction Modulo: {hyperparams.get('reduction_modulo','N/A')}")
                                st.write(f"• Profile: {hyperparams.get('profile','N/A')}")
                            with colB:
                                st.write("**Training Parameters:**")
                                st.write(f"• Epochs: {hyperparams.get('epochs','N/A')}")
                                st.write(f"• Batch Size: {hyperparams.get('batch_size','N/A')}")
                                st.write(f"• Learning Rate: {hyperparams.get('learning_rate','N/A')}")
                                st.write(f"• Error Threshold: {hyperparams.get('error_threshold_percentile','N/A')}%")

                    # Show training metrics if completed
                    if status == "completed" and "metrics" in model_doc:
                        with st.expander("Training Metrics", expanded=True):
                            metrics = model_doc["metrics"]
                            if "loss" in metrics and "val_loss" in metrics:
                                metrics_df = pd.DataFrame({
                                    "epoch": list(range(1, len(metrics["loss"]) + 1)),
                                    "loss": metrics["loss"],
                                    "val_loss": metrics["val_loss"],
                                })
                                fig = px.line(
                                    metrics_df,
                                    x="epoch",
                                    y=["loss", "val_loss"],
                                    labels={"value": "Loss", "variable": "Metric"},
                                    title="Training and Validation Loss",
                                    template="ggplot2",
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    # Actions for completed model
                    if status == "completed":
                        colA, colB, colC, colD = st.columns(4)

                        # Use model
                        with colA:
                            if st.button("🔍 Use This Model"):
                                with st.spinner("Loading model..."):
                                    model_obj, scaler_obj, _ = load_model_from_db(
                                        st.session_state.active_model_id
                                    )
                                    st.session_state.model = model_obj
                                    st.session_state.scaler = scaler_obj
                                    st.session_state.ERROR_THRESHOLD = model_doc.get(
                                        "error_threshold", 0
                                    )
                                    st.session_state.hyperparameters = model_doc.get(
                                        "hyperparameters", {}
                                    )
                                    st.session_state.source_collection = model_doc.get(
                                        "source_collection", None
                                    )

                                    # If source collection is available, load
                                    if st.session_state.source_collection:
                                        st.session_state.df = load_data_from_db(
                                            st.session_state.source_collection
                                        )
                                        if st.session_state.df is not None:
                                            TIMESTAMP = "ts"
                                            ID_COL = "device"
                                            st.session_state.CONTEXT = [TIMESTAMP, ID_COL]
                                            st.session_state.df[ID_COL] = (
                                                st.session_state.df[ID_COL].astype(str)
                                            )
                                            st.session_state.FEATURES = [
                                                col
                                                for col in st.session_state.df.columns
                                                if col not in st.session_state.CONTEXT
                                                and col != "_id"
                                            ]
                                            st.session_state.df[st.session_state.FEATURES] = (
                                                st.session_state.df[st.session_state.FEATURES]
                                                .apply(lambda x: pd.to_numeric(x, errors="coerce"))
                                                .fillna(0.0)
                                            )
                                            st.session_state.df = st.session_state.df.sort_values(
                                                [TIMESTAMP, ID_COL]
                                            ).reset_index(drop=True)
                                            st.session_state.df_copy = st.session_state.df.copy()

                                st.success(f"Model '{model_doc['name']}' loaded successfully!")
                                st.info("Navigate to the Results & Visualization or Anomalies tabs to analyze data.")

                        # Retrain model
                        with colB:
                            if st.button("🔄 Retrain Model"):
                                st.session_state.active_model_id = "new"
                                st.rerun()

                        # Download model
                        with colC:
                            import io
                            if st.button("⏬ Download Model"):
                                with st.spinner("Preparing model for download..."):
                                    model_obj, scaler_obj, _ = load_model_from_db(
                                        st.session_state.active_model_id
                                    )
                                    # 1) Write model to a temporary file
                                    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                                        tmp_name = tmp.name  # This is the actual file path
                                    model_obj.save(tmp_name)  # Writes an H5 file to disk

                                    # 2) Read that file back into memory
                                    with open(tmp_name, "rb") as f:
                                        file_bytes = f.read()

                                    # 3) Delete the temp file
                                    os.remove(tmp_name)

                                # 4) Use download_button to offer the .h5 file
                                st.download_button(
                                    label="Download .h5",
                                    data=file_bytes,
                                    file_name=f"{model_doc['name']}.h5",
                                    mime="application/octet-stream" )

                        # Delete model
                        with colD:
                            if st.button("🗑️ Delete Model"):
                                if (
                                    st.session_state.active_model_id
                                    == st.session_state.training_job
                                ):
                                    st.error("Cannot delete a model that is training!")
                                else:
                                    delete_model(st.session_state.active_model_id)
                                    st.session_state.active_model_id = None
                                    if (
                                        st.session_state.model
                                        and st.session_state.active_model_id
                                        == st.session_state.training_job
                                    ):
                                        st.session_state.model = None
                                        st.session_state.scaler = None
                                    st.success("Model deleted successfully!")
                                    st.rerun()
                else:
                    st.error(f"Model with ID {st.session_state.active_model_id} not found!")
            except Exception as e:
                st.error(f"Error loading model details: {e}")
        else:
            st.info("Select a model from the list or create a new one.")


# ====================================================
# TAB 1: CONFIGURATION
# ====================================================
with tabs[1]:
    st.subheader("Step 1: Configuration")

    # --- Load Data ---
    st.write("##### 1. Load Data from Database")
    st.info(
        "Select a MongoDB collection containing your IoT telemetry data. (e.g. sensor_data)"
    )
    all_collections = db.list_collection_names()
    data_collections = [
        col
        for col in all_collections
        if col != "experiment_logs" and col != "autoencoder_models"
    ]
    if not data_collections:
        st.error("No data collections found in the database!")
    else:
        selected_collection = st.selectbox(
            "Choose a collection:",
            data_collections,
            help="For example, select 'sensor_data' if that is your CSV-equivalent collection.",
        )
        if st.button("Load Data", key="load_data"):
            with st.spinner("Loading data from database..."):
                data = load_data_from_db(selected_collection)
            if data is not None:
                st.session_state.df = data
                st.session_state.df_copy = data.copy()
                st.session_state.source_collection = selected_collection
                TIMESTAMP = "ts"  # Adjust if timestamp field is named differently.
                ID_COL = "device"  # Adjust if device identifier field is named differently.
                st.session_state.CONTEXT = [TIMESTAMP, ID_COL]
                st.session_state.df[ID_COL] = st.session_state.df[ID_COL].astype(str)
                st.session_state.FEATURES = [
                    col
                    for col in st.session_state.df.columns
                    if col not in st.session_state.CONTEXT and col != "_id"
                ]
                st.session_state.df[st.session_state.FEATURES] = (
                    st.session_state.df[st.session_state.FEATURES]
                    .apply(lambda x: pd.to_numeric(x, errors="coerce"))
                    .fillna(0.0)
                )
                st.session_state.df = (
                    st.session_state.df.sort_values([TIMESTAMP, ID_COL], ascending=True)
                    .reset_index(drop=True)
                )
                st.session_state.df_copy = st.session_state.df.copy()
                st.success("Data loaded and prepared successfully!")

    st.write("---")

    # --- Select Training Devices ---
    if st.session_state.df is not None:
        st.write("##### 2. Select Training Devices")
        st.info(
            "Choose the devices (sensor IDs) to be used for training. If you select none, the full dataset will be used."
        )
        training_devices = st.multiselect(
            "Select devices for training:",
            options=st.session_state.df["device"].unique().tolist(),
            default=st.session_state.df["device"].unique().tolist(),
            help="For example, select devices that are stable for training.",
        )
    else:
        training_devices = []

    st.write("---")

    # --- Info about loading models ---


    # Check if a model is already loaded
    if st.session_state.df is not None and st.session_state.model is not None:
        st.write("##### 3. Model Information")
        st.info(
            "⚠️ For model training and management, please use the 'Models' tab. This provides better tracking and management of your models."
        )
        st.success(
            "✅ A model is currently loaded and ready to use. Navigate to the Results or Anomalies tabs to analyze data."
        )

        # Show currently loaded model details
        if st.session_state.active_model_id:
            try:
                model_doc = read_document(
                    {"_id": ObjectId(st.session_state.active_model_id)},
                    "autoencoder_models",
                )
                if model_doc:
                    st.write(f"**Active Model:** {model_doc['name']}")
                    with st.expander("Model Details"):
                        st.write(
                            f"**Description:** {model_doc.get('description', 'No description')}"
                        )
                        st.write(
                            f"**Source Collection:** {model_doc.get('source_collection', 'Unknown')}"
                        )
                        st.write(
                            f"**Error Threshold:** {model_doc.get('error_threshold', 'Unknown')}"
                        )

                        if "hyperparameters" in model_doc:
                            hyperparams = model_doc["hyperparameters"]
                            st.write(f"**Profile:** {hyperparams.get('profile', 'Custom')}")
                            st.write(
                                f"**Latent Dimension:** {hyperparams.get('latent_dim', 'N/A')}"
                            )
                            st.write(f"**Epochs:** {hyperparams.get('epochs', 'N/A')}")
            except:
                st.warning("Could not load active model details.")
    else:
        st.warning(
            "No model is currently loaded. Go to the 'Models' tab to select or create a model."
        )


# ====================================================
# TAB 2: DATA EXPLORATION
# ====================================================
with tabs[2]:
    st.subheader("Step 2: Data Exploration")
    st.info(
        "Here you can review the loaded data. This step gives you an overview and a preview of your dataset."
    )

    if not st.session_state.active_model_id:
        st.warning("No model selected. Go to Tab 0.")
    elif st.session_state.df is None:
        st.warning("No dataset loaded. Go to Tab 1.")
    else:
        st.write(f"Using Model: {st.session_state.active_model_id}")
        st.write(f"Dataset size: {len(st.session_state.df)} rows")
        st.dataframe(st.session_state.df.head(10))
        with st.spinner("Gathering dataset overview..."):
            buf = io.StringIO()
            st.session_state.df.info(buf=buf)
            df_info_str = buf.getvalue()
        st.write("**Dataset Overview:**")
        st.text(df_info_str)
        st.write("**Data Preview (first 50 rows):**")
        st.dataframe(st.session_state.df.head(50))
        st.info("Select a sensor column to view its distribution (histogram).")
        numeric_cols = st.session_state.FEATURES
        chosen_col = st.selectbox(
            "Select a column for distribution:",
            numeric_cols,
            help="For example, choose 'temperature' to see its distribution.",
        )
        if chosen_col:
            with st.spinner("Plotting histogram..."):
                fig_dist = px.histogram(
                    st.session_state.df,
                    x=chosen_col,
                    nbins=50,
                    title=f"Distribution of {chosen_col}",
                    template="ggplot2",
                )
            st.plotly_chart(fig_dist, use_container_width=True)


# ====================================================
# TAB 3: RESULTS & VISUALIZATION
# ====================================================
with tabs[3]:
    st.subheader("Step 3: Results & Visualization")
    st.info(
        "This section displays the training history, loss curves, latent space visualization, and reconstruction error histogram."
    )

    # 1) Check for model and data
    if not st.session_state.active_model_id:
        st.warning(
            "No model is selected. Please go to the Models tab to select or create a model."
        )
    elif st.session_state.df is None:
        st.error("No dataset loaded. Please load data from the Configuration tab first.")
    else:
        # 2) Load model from DB
        try:
            model_obj, scaler_obj, model_info = load_model_from_db(st.session_state.active_model_id)
        except Exception as e:
            st.error(f"Failed to load model from DB: {e}")
            st.stop()

        # 3) Store them in session state for consistency
        st.session_state.model = model_obj
        st.session_state.scaler = scaler_obj

        model_name = model_info["name"] if model_info else "Custom Model"
        st.write("**Using Model:**", model_name)

        # 4) Show training metrics if available
        if model_info and "metrics" in model_info:
            metrics = model_info["metrics"]
            if "loss" in metrics and "val_loss" in metrics:
                st.write("**Training History:**")
                # Convert metrics to a DataFrame
                history_df = pd.DataFrame({
                    "epoch": range(1, len(metrics["loss"]) + 1),
                    "loss": metrics["loss"],
                    "val_loss": metrics["val_loss"],
                })
                st.dataframe(history_df.head())

                # Plot loss curves
                with st.spinner("Plotting loss curves..."):
                    fig_loss = px.line(
                        history_df,
                        x="epoch",
                        y=["loss", "val_loss"],
                        labels={"value": "Loss", "variable": "Metric"},
                        title="Training and Validation Loss",
                        template="ggplot2",
                    )
                st.plotly_chart(fig_loss, use_container_width=True)

        # 5) Scale the current data
        df_scaled = st.session_state.df.copy()
        if st.session_state.FEATURES:
            with st.spinner("Scaling data..."):
                df_scaled[st.session_state.FEATURES] = st.session_state.scaler.transform(
                    df_scaled[st.session_state.FEATURES]
                )

        # 6) Calculate reconstruction errors using the loaded model
        with st.spinner("Calculating reconstruction errors..."):
            df_scaled, preds = AE_tools.calculate_error(
                auto_encoder=st.session_state.model,
                data_frame=df_scaled,
                features=st.session_state.FEATURES,
            )
        st.session_state.df_copy = df_scaled.copy()

        # 7) Build latent space representation
        with st.spinner("Building latent space..."):
            latent_space_df = AE_tools.build_latent_space(
                auto_encoder=st.session_state.model,
                data_frame=df_scaled,
                features=st.session_state.FEATURES,
            )
        st.session_state.latent_space = latent_space_df

        # Merge latent space columns back into df_copy
        for col in latent_space_df.columns:
            st.session_state.df_copy[col] = latent_space_df[col]

        # 8) Determine anomaly threshold from model doc or fallback
        threshold_val = model_info.get("error_threshold") if model_info else None
        if not threshold_val:
            threshold_val = st.session_state.ERROR_THRESHOLD or 0.0
        st.write(f"**Anomaly Threshold:** {threshold_val}")

        # 9) Plot reconstruction error distribution
        with st.spinner("Plotting reconstruction error distribution..."):
            fig_dist_err = px.histogram(
                st.session_state.df_copy,
                x="reconstruction_error",
                nbins=100,
                title="Distribution of Reconstruction Errors",
                template="ggplot2",
            )
            fig_dist_err.add_vline(
                x=threshold_val,
                line_dash="dash",
                line_color="red",
                annotation_text="Anomaly Threshold",
                annotation_position="top right",
            )
        st.plotly_chart(fig_dist_err, use_container_width=True)

        # 10) Latent space visualization (if 2 or more dims)
        if st.session_state.latent_space.shape[1] >= 2:
            st.write("**Latent Space Visualization:**")
            color_overlay = st.selectbox(
                "Select color overlay for latent space:",
                options=[st.session_state.CONTEXT[1], "reconstruction_error"] + st.session_state.FEATURES,
                help="E.g., select 'reconstruction_error' to see how errors vary in latent space.",
            )
            with st.spinner("Plotting latent space..."):
                fig_lat = AE_tools.plot_latent_space(
                    data_frame=st.session_state.df_copy,
                    color=color_overlay,
                    hover_data=st.session_state.CONTEXT,
                    title=f"Latent Space with {color_overlay} Overlay",
                    template="ggplot2",
                )
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("Latent space has fewer than 2 dimensions; skipping visualization.")

# ====================================================
# TAB 4: ANOMALIES
# ====================================================
with tabs[4]:
    st.subheader("Step 4: Anomalies")
    st.info(
        "This section shows anomalies detected based on the reconstruction error. You can also view error trends over time for selected devices."
    )

    if not st.session_state.active_model_id:
        st.warning(
            "No model is loaded. Please go to the Models tab to select or create a model."
        )
    elif st.session_state.df is None:
        st.warning("No dataset loaded. Go to the Configuration tab.")
    elif (
        st.session_state.df_copy is None
        or "reconstruction_error" not in st.session_state.df_copy.columns
    ):
        st.error("Please process the data in the Results & Visualization tab first.")
    else:
        # Get the threshold value
        threshold_val = None
        if st.session_state.active_model_id:
            try:
                model_info = read_document(
                    {"_id": ObjectId(st.session_state.active_model_id)},
                    "autoencoder_models",
                )
                threshold_val = model_info.get("error_threshold")
            except:
                pass

        if threshold_val is None:
            threshold_val = st.session_state.ERROR_THRESHOLD

        st.write(f"**Anomaly Threshold:** {threshold_val}")

        # Find anomalies
        anomalies = st.session_state.df_copy[
            st.session_state.df_copy["reconstruction_error"] > threshold_val
        ]
        st.write(
            f"**Number of Anomalies:** {anomalies.shape[0]} out of {st.session_state.df_copy.shape[0]} records"
        )

        if not anomalies.empty:
            st.write(
                f"**Percentage Anomalous:** {round((anomalies.shape[0] / st.session_state.df_copy.shape[0]) * 100, 2)}%"
            )

            # Add filter options
            st.write("**Filter Anomalies:**")
            col1, col2 = st.columns(2)

            with col1:
                sort_options = st.radio(
                    "Sort by:",
                    [
                        "Highest Error First",
                        "Lowest Error First",
                        "Newest First",
                        "Oldest First",
                    ],
                )

            with col2:
                max_records = st.number_input(
                    "Maximum records to display:",
                    min_value=10,
                    max_value=1000,
                    value=50,
                    step=10,
                )

            # Sort the anomalies based on selection
            if sort_options == "Highest Error First":
                anomalies = anomalies.sort_values("reconstruction_error", ascending=False)
            elif sort_options == "Lowest Error First":
                anomalies = anomalies.sort_values("reconstruction_error", ascending=True)
            elif sort_options == "Newest First":
                anomalies = anomalies.sort_values(
                    st.session_state.CONTEXT[0], ascending=False
                )
            elif sort_options == "Oldest First":
                anomalies = anomalies.sort_values(
                    st.session_state.CONTEXT[0], ascending=True
                )

            # Display anomalies table
            st.write("**Anomalies:**")
            st.dataframe(anomalies.head(max_records))

            # Option to download anomalies as CSV
            csv = anomalies.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="anomalies.csv">Download Anomalies CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Device error visualization
        st.write("**Error Over Time Visualization:**")
        device_list = st.multiselect(
            "Select device(s) for error-over-time visualization:",
            options=st.session_state.df_copy[st.session_state.CONTEXT[1]].unique().tolist(),
            help="Select one or more devices to view their reconstruction error trends over time.",
        )

        if device_list:
            TIMESTAMP = st.session_state.CONTEXT[0]
            ID_COL = st.session_state.CONTEXT[1]

            for device in device_list:
                device_data = st.session_state.df_copy[
                    st.session_state.df_copy[ID_COL] == device
                ].reset_index(drop=True)

                if device_data.empty:
                    st.warning(f"No data available for device {device}")
                    continue

                title = f"Reconstruction Error Over Time: {device}"
                with st.spinner(f"Plotting error over time for {device}..."):
                    fig_eot = AE_tools.plot_error_over_time(
                        df=device_data,
                        timestamp_col=TIMESTAMP,
                        id_col=ID_COL,
                        title=title,
                        template="ggplot2",
                        color_overlay="reconstruction_error",
                        error_threshold=threshold_val,
                    )
                st.plotly_chart(fig_eot, use_container_width=True)

                # Show anomaly details for this device
                device_anomalies = anomalies[anomalies[ID_COL] == device]
                if not device_anomalies.empty:
                    with st.expander(f"Show {len(device_anomalies)} anomalies for {device}"):
                        st.dataframe(device_anomalies)
        else:
            st.info("Select at least one device to view error trends.")

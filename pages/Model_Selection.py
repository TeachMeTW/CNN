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

# Import ML modules and database utilities
import ml.AE_tools as AE_tools
from ml.AE import AutoEncoder
from db_utils.db import db, create_document, read_documents

# ---------------------------
# Configure Logging & Warnings
# ---------------------------
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)
warnings.filterwarnings('ignore')

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
# (Existing utility functions for model and scaler serialization remain unchanged)
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
    return base64.b64encode(pickle.dumps(scaler)).decode('utf-8')

def load_scaler_from_serialized(scaler_str):
    return pickle.loads(base64.b64decode(scaler_str.encode('utf-8')))

# ---------------------------
# Constants for Training
# ---------------------------
SCALER = "Robust"
MACHINE_TYPE = "mac"
SAMPLE_PERC = 0.5

# ---------------------------
# Create Tabs (4 steps)
# ---------------------------
tabs = st.tabs([
    "Configuration", 
    "Data Exploration", 
    "Results & Visualization", 
    "Anomalies"
])

# ====================================================
# TAB 1: CONFIGURATION (Load Data, Select Training Devices, and Train/Update Model)
# ====================================================
with tabs[0]:
    st.subheader("Step 1: Configuration")
    
    # --- Load Data ---
    st.write("##### 1. Load Data from Database")
    st.info("Select a MongoDB collection containing your IoT telemetry data. (e.g. sensor_data)")
    all_collections = db.list_collection_names()
    data_collections = [col for col in all_collections if col != "experiment_logs"]
    if not data_collections:
        st.error("No data collections found in the database!")
    else:
        selected_collection = st.selectbox(
            "Choose a collection:",
            data_collections,
            help="For example, select 'sensor_data' if that is your CSV-equivalent collection."
        )
        if st.button("Load Data", key="load_data"):
            with st.spinner("Loading data from database..."):
                data = load_data_from_db(selected_collection)
            if data is not None:
                st.session_state.df = data
                st.session_state.df_copy = data.copy()
                st.session_state.source_collection = selected_collection
                TIMESTAMP = "ts"  # Adjust if timestamp field is named differently.
                ID_COL = "device" # Adjust if device identifier field is named differently.
                st.session_state.CONTEXT = [TIMESTAMP, ID_COL]
                st.session_state.df[ID_COL] = st.session_state.df[ID_COL].astype(str)
                st.session_state.FEATURES = [col for col in st.session_state.df.columns if col not in st.session_state.CONTEXT and col != "_id"]
                st.session_state.df[st.session_state.FEATURES] = st.session_state.df[st.session_state.FEATURES].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0.0)
                st.session_state.df = st.session_state.df.sort_values([TIMESTAMP, ID_COL], ascending=True).reset_index(drop=True)
                st.session_state.df_copy = st.session_state.df.copy()
                st.success("Data loaded and prepared successfully!")
    
    st.write("---")
    
    # --- Select Training Devices ---
    if st.session_state.df is not None:
        st.write("##### 2. Select Training Devices")
        st.info("Choose the devices (sensor IDs) to be used for training. If you select none, the full dataset will be used.")
        training_devices = st.multiselect(
            "Select devices for training:",
            options=st.session_state.df["device"].unique().tolist(),
            default=st.session_state.df["device"].unique().tolist(),
            help="For example, select devices that are stable for training."
        )
    else:
        training_devices = []
    
    st.write("---")
    
    # --- Train/Update Model ---
    st.write("##### 3. Train/Update AutoEncoder Model")
    st.info("Adjust the hyperparameters for the AutoEncoder model below. Hover over each slider for details.")
    latent_dim_input = st.slider(
        "Latent Dimension", min_value=2, max_value=20, value=5,
        help="This sets the number of neurons in the bottleneck layer (the compressed representation)."
    )
    dropout_input = st.slider(
        "Dropout (%)", min_value=0.01, max_value=0.9, value=0.1, step=0.01,
        help="The percentage of neurons to drop during training to help prevent overfitting."
    )
    reduction_modulo_input = st.slider(
        "Reduction Modulo", min_value=1, max_value=10, value=2,
        help="A parameter to control layer size reduction. Layers are added only if the neuron count is a multiple of this value."
    )
    epochs_input = st.selectbox(
        "Epochs", options=[500, 1000, 1500, 2000], index=1,
        help="The number of training epochs (iterations over the training data)."
    )
    batch_size_input = st.selectbox(
        "Batch Size", options=[128, 256, 512, 1000], index=0,
        help="The number of samples per gradient update during training."
    )
    learning_rate_input = st.selectbox(
        "Learning Rate", options=[0.1, 0.01, 0.001, 0.0001], index=2,
        help="The learning rate for the optimizer; lower values mean slower, more stable training."
    )
    ERROR_CUTOFF_INPUT = st.slider(
        "Error Threshold (Percentile)", min_value=90.0, max_value=99.999, value=99.97, step=0.001,
        help="This percentile is used to determine the anomaly threshold on reconstruction error. For example, 99.97 means that 99.97% of the training data's reconstruction errors are below the threshold."
    )
    
    if st.button("Train New Model", key="train_model"):
        with st.spinner("Training AutoEncoder..."):
            try:
                if st.session_state.df is None:
                    st.error("Please load data first!")
                else:
                    TIMESTAMP = "ts"
                    ID_COL = "device"
                    # Use the selected training devices if any; otherwise, use the full dataset.
                    if training_devices:
                        train_df = st.session_state.df[st.session_state.df[ID_COL].isin(training_devices)]
                    else:
                        train_df = st.session_state.df.copy()
                    if train_df.empty:
                        st.error("No data available for the selected training devices.")
                    else:
                        train_df = train_df.sample(frac=SAMPLE_PERC, random_state=42)
                        train_df, scaler_obj = AE_tools.prep_data(
                            data_frame=train_df.copy(),
                            features=st.session_state.FEATURES,
                            scaler=SCALER
                        )
                        autoenc = AutoEncoder(
                            data=train_df,
                            features=st.session_state.FEATURES,
                            model_name="my_temp_model.h5",
                            log_dir="logs_temp",
                            random_state=42,
                            machine_type=MACHINE_TYPE,
                            n_trials=1
                        )
                        autoencoder_model = autoenc.buildAutoEncoder(
                            latent_dim=latent_dim_input,
                            reduction_modulo=reduction_modulo_input,
                            dropout_percentage=dropout_input,
                            learning_rate=learning_rate_input
                        )
                        callbacks = autoenc.loggingAutoEncoder(
                            autoencoder=autoencoder_model,
                            batch_size=batch_size_input
                        )
                        history_df = autoenc.fitAutoEncoder(
                            model=autoencoder_model,
                            epochs=epochs_input,
                            batch_size=batch_size_input,
                            callbacks=callbacks
                        )
                        st.session_state.model = autoencoder_model
                        st.session_state.scaler = scaler_obj
                        st.session_state.train_df = train_df
                        st.session_state.history = history_df
                        st.session_state.ERROR_THRESHOLD = ERROR_CUTOFF_INPUT
                        st.session_state.hyperparameters = {
                            "latent_dim": latent_dim_input,
                            "dropout": dropout_input,
                            "reduction_modulo": reduction_modulo_input,
                            "epochs": epochs_input,
                            "batch_size": batch_size_input,
                            "learning_rate": learning_rate_input,
                            "error_threshold_percentile": ERROR_CUTOFF_INPUT
                        }
                        st.success("AutoEncoder model trained successfully!")
            except Exception as ex:
                st.error(f"Training failed: {ex}")

# ====================================================
# TAB 2: DATA EXPLORATION
# ====================================================
with tabs[1]:
    st.subheader("Step 2: Data Exploration")
    st.info("Here you can review the loaded data. This step gives you an overview and a preview of your dataset.")
    if st.session_state.df is not None:
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
            help="For example, choose 'temperature' to see its distribution."
        )
        if chosen_col:
            with st.spinner("Plotting histogram..."):
                fig_dist = px.histogram(
                    st.session_state.df,
                    x=chosen_col,
                    nbins=50,
                    title=f"Distribution of {chosen_col}",
                    template="ggplot2"
                )
            st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Please load data from the Configuration tab first.")

# ====================================================
# TAB 3: RESULTS & VISUALIZATION
# ====================================================
with tabs[2]:
    st.subheader("Step 3: Results & Visualization")
    st.info("This section displays the training history, loss curves, latent space visualization, and reconstruction error histogram.")
    if st.session_state.model is not None and st.session_state.scaler is not None:
        if st.session_state.df is None:
            st.error("Data is not loaded. Please load data from the Configuration tab first.")
        else:
            st.write("**Training History:**")
            if st.session_state.history is not None:
                st.dataframe(st.session_state.history.head())
                with st.spinner("Plotting loss curves..."):
                    fig_loss = AE_tools.plot_loss(
                        history=st.session_state.history,
                        title="Training vs. Validation Loss",
                        template="ggplot2"
                    )
                st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.write("No training history available.")
            df_scaled = st.session_state.df.copy()
            df_scaled[st.session_state.FEATURES] = st.session_state.scaler.transform(st.session_state.df[st.session_state.FEATURES])
            df_scaled, preds = AE_tools.calculate_error(
                auto_encoder=st.session_state.model,
                data_frame=df_scaled,
                features=st.session_state.FEATURES
            )
            st.session_state.df_copy = df_scaled.copy()
            latent_space_df = AE_tools.build_latent_space(
                auto_encoder=st.session_state.model,
                data_frame=df_scaled,
                features=st.session_state.FEATURES
            )
            st.session_state.latent_space = latent_space_df
            for col in latent_space_df.columns:
                st.session_state.df_copy[col] = latent_space_df[col]
            # Use the training set if available, else use the full dataset.
            if st.session_state.train_df is not None:
                train_data = st.session_state.train_df.copy()
            else:
                st.warning("Training data not available. Using full dataset for threshold calculation.")
                train_data = st.session_state.df.copy()
            with st.spinner("Calculating anomaly threshold..."):
                train_recon, threshold_val = AE_tools.calculate_error_threshold(
                    auto_encoder=st.session_state.model,
                    train_data=train_data,
                    features=st.session_state.FEATURES,
                    percentile=st.session_state.ERROR_THRESHOLD
                )
            st.write(f"**Anomaly Threshold:** {threshold_val}")
            with st.spinner("Plotting reconstruction error distribution..."):
                fig_dist_err = px.histogram(
                    st.session_state.df_copy,
                    x="reconstruction_error",
                    nbins=100,
                    title="Distribution of Reconstruction Errors",
                    template="ggplot2"
                )
            st.plotly_chart(fig_dist_err, use_container_width=True)
            if st.session_state.latent_space.shape[1] >= 2:
                st.write("**Latent Space Visualization:**")
                color_overlay = st.selectbox(
                    "Select color overlay for latent space:",
                    options=[st.session_state.CONTEXT[1], "reconstruction_error"] + st.session_state.FEATURES,
                    help="For example, select 'reconstruction_error' to see how errors vary in the latent space."
                )
                with st.spinner("Plotting latent space..."):
                    fig_lat = AE_tools.plot_latent_space(
                        data_frame=st.session_state.df_copy,
                        color=color_overlay,
                        hover_data=st.session_state.CONTEXT,
                        title=f"Latent Space with {color_overlay} Overlay",
                        template="ggplot2"
                    )
                st.plotly_chart(fig_lat, use_container_width=True)
            else:
                st.info("Latent space has fewer than 2 dimensions; skipping visualization.")
    else:
        st.info("Model has not been trained yet. Please train a model in the Configuration tab.")

# ====================================================
# TAB 4: ANOMALIES
# ====================================================
with tabs[3]:
    st.subheader("Step 4: Anomalies")
    st.info("This section shows anomalies detected based on the reconstruction error. You can also view error trends over time for selected devices.")
    if st.session_state.df_copy is not None and "reconstruction_error" in st.session_state.df_copy.columns:
        if st.session_state.train_df is not None:
            train_data = st.session_state.train_df.copy()
        else:
            st.warning("Training data not available. Using full dataset for threshold calculation.")
            train_data = st.session_state.df.copy()
        with st.spinner("Calculating anomaly threshold..."):
            _, threshold_val = AE_tools.calculate_error_threshold(
                auto_encoder=st.session_state.model,
                train_data=train_data,
                features=st.session_state.FEATURES,
                percentile=st.session_state.ERROR_THRESHOLD
            )
        st.write(f"**Anomaly Threshold:** {threshold_val}")
        anomalies = st.session_state.df_copy[st.session_state.df_copy["reconstruction_error"] > threshold_val]
        st.write(f"**Number of Anomalies:** {anomalies.shape[0]} out of {st.session_state.df_copy.shape[0]} records")
        if not anomalies.empty:
            st.write(f"**Percentage Anomalous:** {round((anomalies.shape[0] / st.session_state.df_copy.shape[0]) * 100, 2)}%")
            st.dataframe(anomalies.head(50))
        device_list = st.multiselect(
            "Select device(s) for error-over-time visualization:",
            options=st.session_state.df_copy[st.session_state.CONTEXT[1]].unique().tolist(),
            help="Select one or more devices to view their reconstruction error trends over time."
        )
        if device_list:
            TIMESTAMP = st.session_state.CONTEXT[0]
            ID_COL = st.session_state.CONTEXT[1]
            for device in device_list:
                device_data = st.session_state.df_copy[st.session_state.df_copy[ID_COL] == device].reset_index(drop=True)
                title = f"Reconstruction Error Over Time: {device}"
                with st.spinner(f"Plotting error over time for {device}..."):
                    fig_eot = AE_tools.plot_error_over_time(
                        df=device_data,
                        timestamp_col=TIMESTAMP,
                        id_col=ID_COL,
                        title=title,
                        template="ggplot2",
                        color_overlay="reconstruction_error",
                        error_threshold=threshold_val
                    )
                st.plotly_chart(fig_eot, use_container_width=True)
        else:
            st.info("Select at least one device to view error trends.")
    else:
        st.info("Reconstruction error data not available. Please train a model in the Configuration tab.")


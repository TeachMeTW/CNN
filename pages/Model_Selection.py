import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import joblib
import os
import io
import logging
import warnings
import tensorflow as tf
from datetime import datetime

import ml.AE_tools as AE_tools
from ml.AE import AutoEncoder

# Configure logging and Streamlit
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)
warnings.filterwarnings('ignore')

st.set_page_config(page_title="IoT AutoEncoder Dashboard", layout="wide")
st.title("IoT AutoEncoder Dashboard")

# ---------------------------
# 1) Global / Helper Settings
# ---------------------------
ANALYSIS_NAME = "IoT_calPoly_example_{}"
TEMPLATE = "ggplot2"

# We store some session-level variables (so they persist across tabs)
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

if "best_params" not in st.session_state:
    st.session_state.best_params = None

if "latent_space" not in st.session_state:
    st.session_state.latent_space = None

if "history" not in st.session_state:
    st.session_state.history = None


# ---------------------------------------
# 2) Sidebar / Data Uploader & Parameters
# ---------------------------------------
st.sidebar.header("Data & Parameters")

# Step A: Let user pick a CSV or upload file
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

# Alternatively, let user pick a local path (optional fallback)
default_csv_path = "iot_telemetry_data.csv"
local_path = st.sidebar.text_input("Or specify local CSV path:", default_csv_path)

load_data_button = st.sidebar.button("Load Dataset")

# Some default model parameters
st.sidebar.markdown("---")
st.sidebar.subheader("AutoEncoder Hyperparameters (Override Defaults)")

latent_dim_input = st.sidebar.slider("Latent Dimension", 2, 20, 5)
dropout_input = st.sidebar.slider("Dropout (%)", 0.01, 0.9, 0.1, step=0.01)
reduction_modulo_input = st.sidebar.slider("Reduction Modulo", 1, 10, 2)
epochs_input = st.sidebar.selectbox("Epochs", [500, 1000, 1500, 2000], index=1)
batch_size_input = st.sidebar.selectbox("Batch Size", [128, 256, 512, 1000], index=0)
learning_rate_input = st.sidebar.selectbox("Learning Rate", [0.1, 0.01, 0.001, 0.0001], index=2)

# Anomaly threshold percentile
ERROR_CUTOFF_INPUT = st.sidebar.slider("Error Threshold (Percentile)", 90.0, 99.999, 99.97, step=0.001)

# Press button to "Train/Update Model"
train_button = st.sidebar.button("Train/Update AutoEncoder")

# Additional utility placeholders
SCALER = "Robust"
MACHINE_TYPE = "mac"
SAMPLE_PERC = 0.5


# ---------------------------
# 3) Load the Data
# ---------------------------
def load_data():
    """Loads CSV data either from uploaded file or local path. Returns a DataFrame."""
    if uploaded_file is not None:
        df_ = pd.read_csv(uploaded_file)
    else:
        # fallback if local_path is provided
        if not os.path.exists(local_path):
            st.error(f"File not found: {local_path}")
            return None
        df_ = pd.read_csv(local_path)
    return df_

if load_data_button:
    data = load_data()
    if data is not None:
        st.session_state.df = data
        st.session_state.df_copy = data.copy()
        st.success("Dataset loaded successfully!")
    else:
        st.stop()

# If user has loaded data, proceed. Otherwise, wait.
if st.session_state.df is not None:
    # Basic data setup
    # Convert bool -> float
    b_cols = st.session_state.df.select_dtypes(include='bool').columns.to_list()
    st.session_state.df[b_cols] = st.session_state.df[b_cols].astype(float)
    st.session_state.df_copy[b_cols] = st.session_state.df_copy[b_cols].astype(float)

    # For the example: ID_COL, TIMESTAMP
    TIMESTAMP = "ts"
    ID_COL = "device"
    st.session_state.CONTEXT = [TIMESTAMP, ID_COL]

    # Ensure correct dtypes
    st.session_state.df[ID_COL] = st.session_state.df[ID_COL].astype(str)
    # We'll guess the rest of columns as features except the context
    st.session_state.FEATURES = [c for c in st.session_state.df.columns if c not in st.session_state.CONTEXT]
    st.session_state.df[st.session_state.FEATURES] = st.session_state.df[st.session_state.FEATURES].astype(float)

    # Sort the data
    st.session_state.df = st.session_state.df.sort_values([TIMESTAMP, ID_COL], ascending=True).reset_index(drop=True)
    st.session_state.df_copy = st.session_state.df.copy()

    # Print out info in the console (and optionally capture with StringIO)
    buf = io.StringIO()
    st.session_state.df.info(buf=buf)
    df_info_str = buf.getvalue()


# -------------------------------------
# 4) Layout: Multiple Tabs (Dashboard)
# -------------------------------------
if st.session_state.df is not None:
    tabs = st.tabs(["Data Exploration", "Model Training", "Results & Visualization", "Anomalies"])

    # TAB 1: DATA EXPLORATION
    with tabs[0]:
        st.subheader("Data Exploration")
        st.write("**DataFrame Overview**:")
        st.text(df_info_str)

        st.write("**Data Preview**:")
        st.dataframe(st.session_state.df.head(50))

        # Let user pick a column to visualize distribution
        numeric_cols = st.session_state.FEATURES  # or any numeric column
        chosen_col = st.selectbox("Pick a column to see distribution:", numeric_cols)

        if chosen_col:
            fig_dist = px.histogram(
                st.session_state.df,
                x=chosen_col,
                nbins=50,
                title=f"Distribution of {chosen_col}",
                template=TEMPLATE
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # TAB 2: MODEL TRAINING
    with tabs[1]:
        st.subheader("Model Training")

        st.write("**Hyperparameters from Sidebar**:")
        st.write(f"- Latent Dimension: {latent_dim_input}")
        st.write(f"- Dropout: {dropout_input}")
        st.write(f"- Reduction Modulo: {reduction_modulo_input}")
        st.write(f"- Epochs: {epochs_input}")
        st.write(f"- Batch Size: {batch_size_input}")
        st.write(f"- Learning Rate: {learning_rate_input}")
        st.write(f"- Error Cutoff (Percentile): {ERROR_CUTOFF_INPUT}")

        st.write("**Train/Update** the model using the parameters above by pressing the button in the sidebar.")

        if train_button:
            with st.spinner("Training AutoEncoder..."):
                try:
                    # We filter stable devices to create our "train set" (example logic).
                    stable_devices = ["00:0f:00:70:91:0a", "b8:27:eb:bf:9d:51"]
                    train_df = st.session_state.df[st.session_state.df[ID_COL].isin(stable_devices)]
                    train_df = train_df.sample(frac=SAMPLE_PERC, random_state=42)

                    # Scale data
                    train_df, scaler_obj = AE_tools.prep_data(
                        data_frame=train_df.copy(),
                        features=st.session_state.FEATURES,
                        scaler=SCALER
                    )

                    # Build AE with user-defined hyperparams
                    # We'll override the objective search by directly calling buildAutoEncoder & fitAutoEncoder
                    autoenc = AutoEncoder(
                        data=train_df,
                        features=st.session_state.FEATURES,
                        model_name="my_temp_model.h5",
                        log_dir="logs_temp",
                        random_state=42,
                        machine_type=MACHINE_TYPE,
                        n_trials=1  # We skip the search by just building once
                    )

                    # Instead of 'search()', we build with user-chosen hyperparams:
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

                    # Save to session
                    st.session_state.model = autoencoder_model
                    st.session_state.scaler = scaler_obj
                    st.session_state.train_df = train_df
                    st.session_state.history = history_df
                    st.session_state.ERROR_THRESHOLD = ERROR_CUTOFF_INPUT  # we'll apply percentile logic soon

                    st.success("Model trained successfully!")
                except Exception as ex:
                    st.error(f"Failed to train: {ex}")

    # TAB 3: RESULTS & VISUALIZATION
    with tabs[2]:
        st.subheader("Results & Visualizations")

        if st.session_state.model is not None and st.session_state.scaler is not None:
            st.write("**Training History**:")
            if st.session_state.history is not None:
                st.dataframe(st.session_state.history.head())

                # Plot train vs. val loss
                fig_loss = AE_tools.plot_loss(
                    history=st.session_state.history,
                    title="Train vs Validation Loss",
                    template=TEMPLATE
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.write("No training history found.")

            # Apply the trained model to the full dataset
            # 1) Scale full dataset
            df_scaled = st.session_state.df.copy()
            df_scaled[st.session_state.FEATURES] = st.session_state.scaler.transform(
                df_scaled[st.session_state.FEATURES]
            )

            # 2) Calculate recon error
            df_scaled, preds = AE_tools.calculate_error(
                auto_encoder=st.session_state.model,
                data_frame=df_scaled,
                features=st.session_state.FEATURES
            )

            # By default, we just store the recon error in df_copy
            st.session_state.df_copy["reconstruction_error"] = df_scaled["reconstruction_error"]

            # 3) Build latent space
            latent_space_df = AE_tools.build_latent_space(
                auto_encoder=st.session_state.model,
                data_frame=df_scaled,
                features=st.session_state.FEATURES
            )
            st.session_state.latent_space = latent_space_df
            lat_cols = latent_space_df.columns.to_list()

            # Merge latent space into df_copy
            for col in lat_cols:
                st.session_state.df_copy[col] = latent_space_df[col]

            # 4) Determine anomaly threshold (using the training set or user percentile)
            # If the user gave a percentile, we recalc from train_df
            train_recon, threshold_val = AE_tools.calculate_error_threshold(
                auto_encoder=st.session_state.model,
                train_data=st.session_state.train_df.copy(),
                features=st.session_state.FEATURES,
                percentile=st.session_state.ERROR_THRESHOLD
            )
            st.write(f"**Calculated Anomaly Threshold**: {threshold_val}")

            # Let's see distribution of reconstruction errors in the entire data
            fig_dist_err = px.histogram(
                st.session_state.df_copy,
                x="reconstruction_error",
                nbins=100,
                title="Distribution of Reconstruction Errors (Full Data)",
                template=TEMPLATE
            )
            st.plotly_chart(fig_dist_err, use_container_width=True)

            # Latent space plot
            if len(lat_cols) >= 2:
                st.write("**Latent Space Visualization**")
                color_overlay = st.selectbox(
                    "Select color overlay for latent space plot:",
                    [ID_COL, "reconstruction_error"] + st.session_state.FEATURES
                )

                fig_lat = AE_tools.plot_latent_space(
                    data_frame=st.session_state.df_copy,
                    color=color_overlay,
                    hover_data=st.session_state.CONTEXT,
                    title=f"Latent Space: {color_overlay} Overlay",
                    template=TEMPLATE
                )
                st.plotly_chart(fig_lat, use_container_width=True)
            else:
                st.info("Latent dimension < 2, skipping latent space plot.")

        else:
            st.write("Model not trained yet or no data loaded.")

    # TAB 4: ANOMALIES
    with tabs[3]:
        st.subheader("Anomalies Exploration")

        if st.session_state.df_copy is not None and "reconstruction_error" in st.session_state.df_copy.columns:
            # We'll assume threshold_val is computed in the previous tab
            threshold_val = 0
            if st.session_state.ERROR_THRESHOLD and st.session_state.model is not None:
                _, threshold_val = AE_tools.calculate_error_threshold(
                    auto_encoder=st.session_state.model,
                    train_data=st.session_state.train_df.copy(),
                    features=st.session_state.FEATURES,
                    percentile=st.session_state.ERROR_THRESHOLD
                )

            st.write(f"**Using threshold**: {threshold_val}")

            # Identify anomalies
            anomalies = st.session_state.df_copy[
                st.session_state.df_copy["reconstruction_error"] > threshold_val
            ]
            st.write(
                f"**Number of anomalies**: {anomalies.shape[0]} (out of {st.session_state.df_copy.shape[0]})"
            )
            if anomalies.shape[0] > 0:
                st.write(f"**% Anomalous**: {round((anomalies.shape[0]/st.session_state.df_copy.shape[0])*100, 2)}%")
                st.dataframe(anomalies.head(50))

            # Error over time plots
            device_list = st.multiselect(
                "Select device(s) to plot error-over-time:",
                st.session_state.df_copy[ID_COL].unique().tolist(),
                default=None
            )
            if device_list:
                for sn in device_list:
                    device_data = st.session_state.df_copy[st.session_state.df_copy[ID_COL] == sn].reset_index(drop=True)
                    title = f"Reconstruction Error / Time: {sn}"
                    fig_eot = AE_tools.plot_error_over_time(
                        df=device_data,
                        timestamp_col=TIMESTAMP,
                        id_col=ID_COL,
                        title=title,
                        template=TEMPLATE,
                        color_overlay="reconstruction_error",
                        error_threshold=threshold_val
                    )
                    st.plotly_chart(fig_eot, use_container_width=True)
            else:
                st.info("Select at least one device to plot Error-over-time.")
        else:
            st.write("No reconstruction_error found. Train the model and re-check.")

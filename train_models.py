# ml/train_model.py
import sys
import json
import pickle
import os
import tensorflow as tf
import time
from datetime import datetime
import logging
from bson import ObjectId

# Import your modules
# Fixed: Corrected relative import path
# from ..db_utils.db import db, update_document, read_document
from db_utils.db import db, update_document, read_document  # Absolute import
import ml.AE_tools as AE_tools
from ml.AE import AutoEncoder
from ml.model_helper import serialize_model, serialize_scaler


# Configure logging
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def check_for_cancellation(model_id):
    """Check if training should be cancelled"""
    model_doc = read_document({"_id": ObjectId(model_id)}, "autoencoder_models")
    return model_doc.get("cancel_requested", False)


class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_id, total_epochs):
        super().__init__()
        self.model_id = model_id
        self.total_epochs = total_epochs
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs

        # Update progress in database
        update_document(
            {"_id": ObjectId(self.model_id)},
            {"training_progress": progress},
            "autoencoder_models",
        )

        # Check for cancellation
        if check_for_cancellation(self.model_id):
            logging.info(f"Training cancelled for model {self.model_id}")
            self.model.stop_training = True
            update_document(
                {"_id": ObjectId(self.model_id)},
                {"training_status": "cancelled"},
                "autoencoder_models",
            )

    def on_train_end(self, logs=None):
        if check_for_cancellation(self.model_id):
            logging.info(f"Training cancelled for model {self.model_id}")
            update_document(
                {"_id": ObjectId(self.model_id)},
                {"training_status": "cancelled"},
                "autoencoder_models",
            )


def main():
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <config_file>")
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        model_id = config["model_id"]
        data_path = config["data_path"]

        # Load training data
        with open(data_path, "rb") as f:
            training_data = pickle.load(f)

        df = training_data["df"]
        training_devices = training_data["training_devices"]
        hyperparams = training_data["hyperparams"]

        logging.info(f"Starting training for model {model_id}")

        # Actual training code from your original train_model_async function
        TIMESTAMP = "ts"
        ID_COL = "device"
        SCALER = "Robust"
        MACHINE_TYPE = "mac"
        SAMPLE_PERC = 0.5
        features = [
            col for col in df.columns if col not in [TIMESTAMP, ID_COL, "_id"]
        ]

        # Filter by training devices if specified
        if training_devices:
            train_df = df[df[ID_COL].isin(training_devices)]
        else:
            train_df = df.copy()

        # Sample data
        train_df = train_df.sample(frac=SAMPLE_PERC, random_state=42)

        # Prepare data
        train_df, scaler_obj = AE_tools.prep_data(
            data_frame=train_df.copy(), features=features, scaler=SCALER
        )

        # Build model
        autoenc = AutoEncoder(
            data=train_df,
            features=features,
            model_name=f"temp_model_{model_id}.h5",
            log_dir=f"logs_temp_{model_id}",
            random_state=42,
            machine_type=MACHINE_TYPE,
            n_trials=1,
        )

        autoencoder_model = autoenc.buildAutoEncoder(
            latent_dim=hyperparams["latent_dim"],
            reduction_modulo=hyperparams["reduction_modulo"],
            dropout_percentage=hyperparams["dropout"],
            learning_rate=hyperparams["learning_rate"],
        )

        # Setup callbacks
        std_callbacks = autoenc.loggingAutoEncoder(
            autoencoder=autoencoder_model, batch_size=hyperparams["batch_size"]
        )

        # Add custom progress callback
        progress_callback = TrainingCallback(model_id, hyperparams["epochs"])
        all_callbacks = std_callbacks + [progress_callback]

        # Train model
        history_df = autoenc.fitAutoEncoder(
            model=autoencoder_model,
            epochs=hyperparams["epochs"],
            batch_size=hyperparams["batch_size"],
            callbacks=all_callbacks,
        )

        # Calculate error threshold
        _, threshold_val = AE_tools.calculate_error_threshold(
            auto_encoder=autoencoder_model,
            train_data=train_df,
            features=features,
            percentile=hyperparams["error_threshold_percentile"],
        )

        # Final update to database with trained model
        history_dict = history_df.to_dict("list") if history_df is not None else None

        update_document(
            {"_id": ObjectId(model_id)},
            {
                "model": serialize_model(autoencoder_model),
                "scaler": serialize_scaler(scaler_obj),
                "error_threshold": threshold_val,
                "metrics": {
                    "loss": history_dict.get("loss", []) if history_dict else [],
                    "val_loss": history_dict.get("val_loss", []) if history_dict else [],
                },
                "training_status": "completed",
                "completed_at": datetime.now(),
                "training_time": time.time() - progress_callback.start_time,
            },
            "autoencoder_models",
        )

        logging.info(f"Training completed for model {model_id}")

        # Clean up temporary files
        try:
            os.remove(data_path)
            os.remove(config_path)
        except Exception as e:
            logging.warning(f"Error cleaning up temp files: {e}")

    except Exception as e:
        logging.error(f"Training error: {e}", exc_info=True)

        # Update status to failed
        try:
            update_document(
                {"_id": ObjectId(model_id)},
                {
                    "training_status": "failed",
                    "error_message": str(e),
                    "completed_at": datetime.now(),
                },
                "autoencoder_models",
            )
        except Exception as inner_e:
            logging.error(f"Error updating model status: {inner_e}")


if __name__ == "__main__":
    main()

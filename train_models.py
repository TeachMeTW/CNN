# ml/train_model.py
import sys
import time
import logging
from datetime import datetime
from bson import ObjectId
import tensorflow as tf

# Import your modules
from db_utils.db import read_document, update_document
import ml.AE_tools as AE_tools
from ml.AE import AutoEncoder
from ml.model_helper import serialize_model, serialize_scaler

logging.basicConfig(level=logging.INFO)


def check_for_cancellation(model_id):
    doc = read_document({"_id": ObjectId(model_id)}, "autoencoder_models")
    return doc.get("cancel_requested", False)


class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_id, total_epochs):
        super().__init__()
        self.model_id = model_id
        self.total_epochs = total_epochs
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        update_document(
            {"_id": ObjectId(self.model_id)},
            {"training_progress": progress},
            "autoencoder_models",
        )
        if check_for_cancellation(self.model_id):
            self.model.stop_training = True
            update_document(
                {"_id": ObjectId(self.model_id)},
                {"training_status": "cancelled"},
                "autoencoder_models",
            )

    def on_train_end(self, logs=None):
        if check_for_cancellation(self.model_id):
            update_document(
                {"_id": ObjectId(self.model_id)},
                {"training_status": "cancelled"},
                "autoencoder_models",
            )


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]
    try:
        # Retrieve job from DB
        model_doc = read_document({"_id": ObjectId(model_id)}, "autoencoder_models")
        if not model_doc:
            logging.error("Model doc not found in DB.")
            sys.exit(1)

        # Load config from doc
        source_collection = model_doc["source_collection"]
        hyperparams = model_doc["hyperparameters"]
        training_devices = model_doc["training_devices"]

        # Load data from DB
        from ml.model_helper import load_data_from_db

        df = load_data_from_db(source_collection)
        if training_devices:
            df = df[df["device"].isin(training_devices)]
        df = df.sample(frac=0.5, random_state=42)  # Example sampling

        features = [c for c in df.columns if c not in ["_id", "ts", "device"]]

        # Prepare data
        df_prep, scaler_obj = AE_tools.prep_data(df.copy(), features, scaler="Robust")

        # Build & train model
        autoenc = AutoEncoder(
            data=df_prep,  # Use prepped data
            features=features,
            model_name=f"temp_model_{model_id}.h5",
            log_dir=f"logs_temp_{model_id}",
            random_state=42,
            machine_type="mac",
            n_trials=1,
        )
        model = autoenc.buildAutoEncoder(
            latent_dim=hyperparams["latent_dim"],
            reduction_modulo=hyperparams["reduction_modulo"],
            dropout_percentage=hyperparams["dropout"],
            learning_rate=hyperparams["learning_rate"],
        )

        cb = TrainingCallback(model_id, hyperparams["epochs"])
        callbacks = autoenc.loggingAutoEncoder(model, hyperparams["batch_size"]) + [cb]
        history_df = autoenc.fitAutoEncoder(
            model, hyperparams["epochs"], hyperparams["batch_size"], callbacks
        )

        # Compute error threshold
        _, threshold_val = AE_tools.calculate_error_threshold(
            auto_encoder=model,
            train_data=df_prep,
            features=features,
            percentile=hyperparams["error_threshold_percentile"],
        )

        # Update doc with final model
        update_document(
            {"_id": ObjectId(model_id)},
            {
                "model": serialize_model(model),
                "scaler": serialize_scaler(scaler_obj),
                "error_threshold": threshold_val,
                "metrics": {
                    "loss": history_df["loss"].tolist(),
                    "val_loss": history_df["val_loss"].tolist(),
                },
                "training_status": "completed",
                "training_progress": 1.0,
                "completed_at": datetime.now(),
            },
            "autoencoder_models",
        )
        logging.info(f"Training completed for model {model_id}.")

    except Exception as e:
        logging.error(f"Training error: {e}", exc_info=True)
        update_document(
            {"_id": ObjectId(model_id)},
            {"training_status": "failed", "error_message": str(e)},
            "autoencoder_models",
        )


if __name__ == "__main__":
    main()

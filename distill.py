import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TFDS_DATA_DIR"] = "/data/audio/"
os.environ["TFHUB_CACHE_DIR"] = "/cached_models/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import larq as lq

from model import BinaryModel, Distiller

parser = argparse.ArgumentParser(description="")
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--dataset", type=constants.Dataset, default=constants.Dataset.SPC)
parser.add_argument("--model_directory", type=str, default="./saved_models")
parser.add_argument("--distilled_model_directory", type=str, default="./distilled_models")
parser.add_argument("--student", default="quicknet", type=str)
parser.add_argument("--teacher", default="trillsson_3", type=str)
args = parser.parse_args()

def create_pretrained_trillsson_model(id=3):
    trillsson = hub.KerasLayer(
        f"https://tfhub.dev/google/trillsson{id}/1", 
        trainable=False)
    inp = tf.keras.layers.Input((None,))
    out = trillsson(inp)["embedding"]
    model = tf.keras.Model(inp, out)
    return model

def run_distillation(args):
    print("\n".join(["{}: {}".format(k, v) for k, v in vars(args).items()]), flush=True)

    data_dir = os.getenv("TFDS_DATA_DIR")

    print("Creating teacher model...", flush=True)
    teacher_model = create_pretrained_trillsson_model() 
    hidden_size = 1024

    input_shape = (None,)
    print("Creating data builder...", flush=True)
    # Implement data loading...
    train_dataset = ...

    print("Creating student model...", flush=True)
    audio_model = BinaryModel()
    student_model = audio_model.create_model(
        input_shape, 0, args.student, hidden_size)
    lq.models.summary(student_model)

    print("Setting up distiller...", flush=True)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    distiller = Distiller(student_model, teacher_model)
    distiller.compile(
        optimizer=optimizer,
        distillation_loss_fn=tf.keras.losses.MeanSquaredError()) 
    
    checkpoint_filepath = f"{args.model_directory}/{args.teacher}_{args.student}.h5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_best_only=False)

    print("Training model...", flush=True)
    distiller.fit(train_dataset, 
        epochs=args.epochs, 
        callbacks=[model_checkpoint_callback], 
        verbose=2)

    print("Saving weights...", flush=True)
    pretrained_model = distiller.student
    pretrained_model.save_weights(
        f"{args.distilled_model_directory}/{args.student}_weights.h5")

if __name__ == "__main__":
   run_distillation(args)

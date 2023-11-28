import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TFDS_DATA_DIR"] = "/data/audio/"
os.environ["TFHUB_CACHE_DIR"] = "/cached_models"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import larq as lq
from sklearn import metrics
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa

from model import BinaryModel

parser = argparse.ArgumentParser(description="")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--dataset", default="speech_commands", type=str)
parser.add_argument("--student", default="densenet", type=str)
parser.add_argument("--model_directory", default="./saved_models", type=str)
parser.add_argument("--distilled_model_directory", default="./distilled_models", type=str)
args = parser.parse_args()

def prepare_example(waveform, label, sequence_length=16000):
  waveform = tf.cast(waveform, tf.float32) / float(tf.int16.max)
  padding = tf.maximum(sequence_length - tf.shape(waveform)[0], 0)
  left_pad = padding // 2
  right_pad = padding - left_pad
  waveform = tf.pad(waveform, paddings=[[left_pad, right_pad]])
  return waveform, label

def run_evaluation(args):
    print("Loading dataset...", flush=True)
    autotune = tf.data.AUTOTUNE
    (ds_train, ds_val, ds_test), ds_info = tfds.load(dataset, 
        split=["train", "validation", "test"], shuffle_files=True, 
        as_supervised=True, with_info=True)
    num_classes =  ds_info.features["label"].num_classes

    ds_train = ds_train.map(prepare_example, num_parallel_calls=autotune)
    ds_train = ds_train.batch(batch_size).prefetch(autotune)

    ds_test = ds_test.map(prepare_example, num_parallel_calls=autotune)
    ds_test = ds_test.batch(batch_size).prefetch(autotune)

    print("Creating pretrained model...")
    input_shape = (None,)
    hidden_size = 1024
    audio_model = BinaryModel()
    pretrained_model = audio_model.create_model(
        input_shape, None, args.student, hidden_size)
    pretrained_model.load_weights(f"{args.distilled_model_directory}/{args.student}_weights.h5")
    pretrained_model = pretrained_model.get_layer("model")
    pretrained_model.trainable = False

    print("Creating classification model...")
    clf_model = audio_model.create_classification_model(input_shape, 
        pretrained_model, num_classes, args.augment)
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 
    lq.models.summary(clf_model)    

    clf_model.fit(train_dataset, epochs=args.epochs, verbose=2)
    clf_model.evaluate(test_dataset, verbose=2)

    with lq.context.quantized_scope(True):
        weights = clf_model.get_weights()
        clf_model.set_weights(weights)

    tflite_model = lce.convert_keras_model(clf_model)
    with open(f"./{args.model_directory}/{args.dataset}_{args.student}_dist_clf.tflite", "wb") as f:
        f.write(tflite_model)

if __name__ == "__main__":
    run_evaluation(args)
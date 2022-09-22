import os
import random
import numpy as np
import pandas as pd

import warnings

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow_addons as tfa

from transformers import AutoTokenizer
from transformers import TFAutoModel, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)

from config import parser
from data_loader import DataGenerator
from model import image_model, text_model, mutlitmodal_model

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.join(current_dir, os.pardir)
args = parser()

optimizer = args.optimizer
learning_rate = args.learning_rate
loss = args.loss
label_smoothing = args.label_smoothing
BATCH_SIZE = args.batch_size
IMAGE_SIZE = args.image_size
EPOCHS = args.epochs
validation_size = args.validation_size
MAX_LENGTH = args.max_length
seed = args.seed

lr = tf.keras.optimizers.schedules.CosineDecay(learning_rate, decay_steps=1000)

if args.text_pretrained_model == "kosroberta":
    text_pretrained_model = "jhgan/ko-sroberta-multitask"
elif args.text_pretrained_model == "koelectra":
    text_pretrained_model = "monologg/koelectra-base-v3-discriminator"
elif args.text_pretrained_model == "roberta":
    text_pretrained_model = "klue/roberta-large"
elif args.text_pretrained_model == "electra":
    text_pretrained_model = "kykim/electra-kor-base"
elif args.text_pretrained_model == "funnel":
    text_pretrained_model = "kykim/funnel-kor-base"

if args.image_pretrained_model == "InceptionV3":
    image_pretrained_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_tensor=layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    )

if args.optimizer == "sgd":
    optim = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
elif args.optimizer == "adam":
    optim = tf.keras.optimizers.Adam(learning_rate=lr)

if loss == "cc":
    loss_function = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing
    )
elif loss == "fl":
    loss_function = tfa.losses.SigmoidFocalCrossEntropy()

tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model)
pretrained_bert = TFAutoModel.from_pretrained(text_pretrained_model, from_pt=True)

def set_seeds(seed=seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    
    os.chdir(os.path.join(upper_dir, "data"))

    set_seeds()

    train_df = pd.read_csv(os.path.join("pp_train.csv"))
    test_df = pd.read_csv("./pp_test.csv")

    le = LabelEncoder()
    le.fit(train_df["cat3"].values)

    NUM_CLASS = len(le.classes_)
    train_df["cat3"] = le.transform(train_df["cat3"].values)

    X_test = test_df[["img_path", "overview"]]
    X, y = train_df[["img_path", "overview"]], tf.keras.utils.to_categorical(train_df["cat3"])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, random_state=seed, stratify=y
    )

    train_ds = DataGenerator(
        img_path_list = X_train['img_path'].values,
        img_size = IMAGE_SIZE,
        sentence = X_train['overview'].values,
        max_length = MAX_LENGTH,
        tokenizer = tokenizer,
        labels = y_train,
        batch_size=BATCH_SIZE,
        seed = seed,
        shuffle=True,
        include_targets=True,
    )

    val_ds = DataGenerator(
        img_path_list = X_val['img_path'].values,
        img_size = IMAGE_SIZE,
        sentence = X_val['overview'].values,
        max_length = MAX_LENGTH,
        tokenizer = tokenizer,
        labels = y_val,
        batch_size=BATCH_SIZE,
        seed = seed,
        shuffle=True,
        include_targets=True,
    )

    img_side = image_model(image_pretrained_model)
    text_side = text_model(pretrained_bert, MAX_LENGTH)
    multi_side = mutlitmodal_model(img_side, text_side, IMAGE_SIZE, MAX_LENGTH, NUM_CLASS)

    print(img_side.summary())
    print(text_side.summary())
    print(multi_side.summary())

    multi_side.compile(
        optimizer=optim,
        loss=loss_function,
        metrics=[
            tfa.metrics.F1Score(
                num_classes=len(le.classes_),
                average="weighted",
            ),
        ],
    )

    checkpoint_callback = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "../MULTI_PRETRAINED",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    history = multi_side.fit(
        train_ds,
        validation_data=val_ds,
        verbose=True,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback],
    )

if __name__ == "__main__":

    main()
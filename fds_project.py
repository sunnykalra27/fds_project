import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def clean_text(text: str) -> str:
    text = str(text)
    text = text.lower()
    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # remove non-letters (keep basic punctuation)
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_dataframe(df, min_samples=30):
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["venue"] = df["venue"].astype(str)

    # Combine title + description and clean
    df["text_raw"] = df["title"] + " " + df["description"]
    df["text"] = df["text_raw"].apply(clean_text)
    venue_counts = df["venue"].value_counts()
    valid_venues = venue_counts[venue_counts >= min_samples].index
    df_filtered = df[df["venue"].isin(valid_venues)].copy()
    label_encoder = LabelEncoder()
    df_filtered["label"] = label_encoder.fit_transform(df_filtered["venue"])
    return df_filtered, label_encoder

def split_data(df, test_size=0.2, random_state=42):
    X = df["text"].values
    y = df["label"].values

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

def build_bilstm_model(
    vocab_size,
    max_len,
    num_classes,
    embedding_dim=128
):
    inputs = layers.Input(shape=(max_len,), dtype="int32")

    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = layers.SpatialDropout1D(0.3)(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True)
    )(x)

    x_max = layers.GlobalMaxPooling1D()(x)
    x_avg = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([x_max, x_avg])

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_class_weights(y_train):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    return {i: float(w) for i, w in enumerate(weights)}

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights,
    batch_size=64,
    epochs=25
):
    es = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    rlr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[es, rlr],
        verbose=1
    )

    return history

def plot_history(history):
    plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.legend()
    plt.title("Accuracy")
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend()
    plt.title("Loss")
    plt.show()

def predict_venue(
    text,
    model,
    tokenizer,
    label_encoder,
    max_len
):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=max_len, padding="post")

    probs = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(probs))

    return (
        label_encoder.inverse_transform([idx])[0],
        float(probs[idx])
    )















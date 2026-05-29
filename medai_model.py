"""
MedAI Model — Custom TensorFlow/Keras NLP model
Trained on your own medical text/PDF data.

Given a disease name (or symptoms), predicts:
  - Medicine / tablet names
  - Side effects
  - Alternative medicines
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


# ─── Config ────────────────────────────────────────────────────────────────────
MAX_VOCAB       = 5000
MAX_SEQ_LEN     = 20
EMBED_DIM       = 64
HIDDEN_UNITS    = 128
EPOCHS          = 60
BATCH_SIZE      = 8
MODEL_DIR       = "models"
# ───────────────────────────────────────────────────────────────────────────────


class MedAIModel:

    def __init__(self):
        self.tokenizer      = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
        self.mlb_medicines  = MultiLabelBinarizer()
        self.mlb_effects    = MultiLabelBinarizer()
        self.mlb_alts       = MultiLabelBinarizer()
        self.model          = None
        self.history        = None

    # ── 1. Prepare Data ─────────────────────────────────────────────────────────

    def prepare_data(self, records):
        """Convert parsed records into model-ready tensors."""
        diseases   = [r["disease"] for r in records]
        medicines  = [r.get("medicines",   []) for r in records]
        effects    = [r.get("side_effects",[]) for r in records]
        alts       = [r.get("alternatives",[]) for r in records]

        # Fit tokenizer on disease names
        self.tokenizer.fit_on_texts(diseases)
        X = self.tokenizer.texts_to_sequences(diseases)
        X = pad_sequences(X, maxlen=MAX_SEQ_LEN, padding="post")

        # Fit multi-label binarizers
        Y_med  = self.mlb_medicines.fit_transform(medicines)
        Y_eff  = self.mlb_effects.fit_transform(effects)
        Y_alt  = self.mlb_alts.fit_transform(alts)

        print(f"Input shape      : {X.shape}")
        print(f"Medicines labels : {Y_med.shape[1]}")
        print(f"Side-effects lbl : {Y_eff.shape[1]}")
        print(f"Alternatives lbl : {Y_alt.shape[1]}")

        return X, Y_med, Y_eff, Y_alt

    # ── 2. Build Model ───────────────────────────────────────────────────────────

    def build_model(self, n_medicines, n_effects, n_alts):
        """
        Multi-output model:
          Input  →  Embedding  →  BiLSTM  →  Dense
                                           ├─ output_medicines
                                           ├─ output_side_effects
                                           └─ output_alternatives
        """
        inp = layers.Input(shape=(MAX_SEQ_LEN,), name="disease_input")

        x = layers.Embedding(MAX_VOCAB, EMBED_DIM, name="embedding")(inp)
        x = layers.Bidirectional(
                layers.LSTM(HIDDEN_UNITS, return_sequences=True), name="bilstm"
            )(x)
        x = layers.GlobalAveragePooling1D(name="pool")(x)
        x = layers.Dense(256, activation="relu", name="shared_dense")(x)
        x = layers.Dropout(0.3)(x)

        out_med  = layers.Dense(n_medicines, activation="sigmoid", name="medicines")(x)
        out_eff  = layers.Dense(n_effects,   activation="sigmoid", name="side_effects")(x)
        out_alt  = layers.Dense(n_alts,      activation="sigmoid", name="alternatives")(x)

        self.model = Model(inputs=inp, outputs=[out_med, out_eff, out_alt])

        self.model.compile(
            optimizer="adam",
            loss={
                "medicines":   "binary_crossentropy",
                "side_effects":"binary_crossentropy",
                "alternatives":"binary_crossentropy",
            },
            metrics={"medicines": "accuracy", "side_effects": "accuracy", "alternatives": "accuracy"},
        )
        self.model.summary()
        return self.model

    # ── 3. Train ─────────────────────────────────────────────────────────────────

    def train(self, records):
        """Full training pipeline."""
        print("\n── Preparing data ──────────────────────────────")
        X, Y_med, Y_eff, Y_alt = self.prepare_data(records)

        # Train/val split
        idx = np.arange(len(X))
        if len(idx) > 4:
            tr, val = train_test_split(idx, test_size=0.2, random_state=42)
        else:
            tr = val = idx  # too small — train on all

        print("\n── Building model ──────────────────────────────")
        self.build_model(Y_med.shape[1], Y_eff.shape[1], Y_alt.shape[1])

        print("\n── Training ────────────────────────────────────")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        ]

        self.history = self.model.fit(
            X[tr],
            {"medicines": Y_med[tr], "side_effects": Y_eff[tr], "alternatives": Y_alt[tr]},
            validation_data=(
                X[val],
                {"medicines": Y_med[val], "side_effects": Y_eff[val], "alternatives": Y_alt[val]},
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )
        print("\n── Training complete ───────────────────────────")
        return self.history

    # ── 4. Predict ───────────────────────────────────────────────────────────────

    def predict(self, disease_name, threshold=0.3, top_k=5):
        """Given a disease name, return medicines, side effects, and alternatives."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        seq = self.tokenizer.texts_to_sequences([disease_name.lower()])
        seq = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post")

        pred_med, pred_eff, pred_alt = self.model.predict(seq, verbose=0)

        def decode(probs, binarizer, threshold, top_k):
            classes = binarizer.classes_
            scored  = sorted(zip(classes, probs[0]), key=lambda x: -x[1])
            above   = [(c, float(p)) for c, p in scored if p >= threshold]
            if not above:
                above = scored[:3]     # fallback: always return top-3
            return above[:top_k]

        results = {
            "disease":      disease_name,
            "medicines":    decode(pred_med, self.mlb_medicines,  threshold, top_k),
            "side_effects": decode(pred_eff, self.mlb_effects,    threshold, top_k),
            "alternatives": decode(pred_alt, self.mlb_alts,       threshold, top_k),
        }
        return results

    # ── 5. Save / Load ───────────────────────────────────────────────────────────

    def save(self, directory=MODEL_DIR):
        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, "medai_model.h5"))
        with open(os.path.join(directory, "tokenizer.pkl"),    "wb") as f: pickle.dump(self.tokenizer,     f)
        with open(os.path.join(directory, "mlb_medicines.pkl"),"wb") as f: pickle.dump(self.mlb_medicines,  f)
        with open(os.path.join(directory, "mlb_effects.pkl"),  "wb") as f: pickle.dump(self.mlb_effects,    f)
        with open(os.path.join(directory, "mlb_alts.pkl"),     "wb") as f: pickle.dump(self.mlb_alts,       f)
        print(f"Model saved → {directory}/")

    def load(self, directory=MODEL_DIR):
        self.model = tf.keras.models.load_model(os.path.join(directory, "medai_model.h5"))
        with open(os.path.join(directory, "tokenizer.pkl"),    "rb") as f: self.tokenizer     = pickle.load(f)
        with open(os.path.join(directory, "mlb_medicines.pkl"),"rb") as f: self.mlb_medicines  = pickle.load(f)
        with open(os.path.join(directory, "mlb_effects.pkl"),  "rb") as f: self.mlb_effects    = pickle.load(f)
        with open(os.path.join(directory, "mlb_alts.pkl"),     "rb") as f: self.mlb_alts       = pickle.load(f)
        print(f"Model loaded from {directory}/")

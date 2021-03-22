import config
import tensorflow as tf
from transformers import TFBertModel
from tensorflow_addons import metrics
from encoding import BertPreprocessing


def train_model(train_df, valid_df, y_train, y_valid):
    input_ids = tf.keras.layers.Input(
        shape=(config.max_length,), dtype=tf.int32, name="input_ids"
    )
    attention_masks = tf.keras.layers.Input(
        shape=(config.max_length,), dtype=tf.int32, name="attention_masks"
    )
    token_type_ids = tf.keras.layers.Input(
        shape=(config.max_length,), dtype=tf.int32, name="token_type_ids"
    )

    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    bert_model.trainable = False

    out = bert_model.bert(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )

    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(out[0])
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_data = BertPreprocessing(
        train_df[["sentence1", "sentence2"]].values.astype("str"),
        y_train,
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_data = BertPreprocessing(
        valid_df[["sentence1", "sentence2"]].values.astype("str"),
        y_valid,
        batch_size=config.batch_size,
        shuffle=False,
    )

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=config.epochs,
        use_multiprocessing=True,
        workers=-1,
    )

    bert_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-5),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
            metrics.F1Score(num_classes=3, average="macro"),
        ],
    )

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=config.epochs,
        use_multiprocessing=True,
        workers=-1,
    )
    return model
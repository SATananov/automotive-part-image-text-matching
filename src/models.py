from __future__ import annotations

from typing import Any

IMAGE_SHAPE = (24, 24, 3)
SEQUENCE_LENGTH = 12
EMBEDDING_SIZE = 16
NUMBER_OF_CLASSES = 3


def build_text_model(keras: Any, vocabulary_size: int) -> Any:
    text = keras.Input(shape=(SEQUENCE_LENGTH,), dtype="int32", name="text")
    x = keras.layers.Embedding(vocabulary_size, EMBEDDING_SIZE)(text)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.10)(x)
    output = keras.layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(x)
    return compile_model(keras.Model(text, output, name="text_model"), keras)


def image_branch(keras: Any, image: Any) -> Any:
    x = keras.layers.Rescaling(1.0 / 255.0)(image)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    return keras.layers.Dense(32, activation="relu")(x)


def build_image_model(keras: Any) -> Any:
    image = keras.Input(shape=IMAGE_SHAPE, name="image")
    x = image_branch(keras, image)
    x = keras.layers.Dropout(0.10)(x)
    output = keras.layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(x)
    return compile_model(keras.Model(image, output, name="image_model"), keras)


def build_multimodal_model(keras: Any, vocabulary_size: int) -> Any:
    text = keras.Input(shape=(SEQUENCE_LENGTH,), dtype="int32", name="text")
    text_features = keras.layers.Embedding(vocabulary_size, EMBEDDING_SIZE)(text)
    text_features = keras.layers.GlobalAveragePooling1D()(text_features)
    text_features = keras.layers.Dense(32, activation="relu")(text_features)

    image = keras.Input(shape=IMAGE_SHAPE, name="image")
    image_features = image_branch(keras, image)

    x = keras.layers.Concatenate()([image_features, text_features])
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.15)(x)
    output = keras.layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(x)
    model = keras.Model({"image": image, "text": text}, output, name="multimodal_model")
    return compile_model(model, keras)


def compile_model(model: Any, keras: Any) -> Any:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

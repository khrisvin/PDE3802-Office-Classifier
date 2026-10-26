from tensorflow import keras

from dataset import classes


def build_model():
    keras.utils.set_random_seed(42)

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3), name="rgb_image")

    x = keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="pixel_scaling"
    )(inputs)

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="feature_average")(x)

    outputs = keras.layers.Dense(
        len(classes),
        activation="softmax",
        name="office_classes"
    )(x)

    model = keras.Model(inputs, outputs, name="office_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary() 
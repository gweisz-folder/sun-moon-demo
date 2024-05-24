# Imports
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import keras_cv
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "tensorflow"

# Behavior flags

#  If true an existent model named MODEL_FILENAME will be loaded
#  Otherwise a new model will be created from scratch leveraging "efficientnetv2_b0_imagenet"
LOAD_MODEL = False

#  If true, the model will be trained
TRAIN_MODEL = True

#  If true, the model will be saved as MODEL_FILENAME
SAVE_MODEL = False

#   If true, the validation data predictions will be shown along with the actual classes
SHOW_PREDS = True

#  Main parameters
DS_NAME = 'sun_moon'
MODEL_FILENAME = './model.keras'
BATCH_SIZE = 4
VALIATION_BATCH_SIZE = 10
IMAGE_TARGET_SIZE = (224, 224)
CLASSES = {0: "sun", 1: "moon"}
LEARNING_RATE = 0.02
EPOCHS_A = 45
EPOCHS_B = 10

# Functions

resizing_f = keras_cv.layers.Resizing(
    IMAGE_TARGET_SIZE[0], IMAGE_TARGET_SIZE[1], crop_to_aspect_ratio=True
)


def preprocess_data(images, labels):
    return resizing_f(images), tf.one_hot(labels, len(CLASSES))


data_augmentation = keras.Sequential(
    [
        keras.layers.RandomRotation(factor=0.1),
        keras.layers.RandomFlip("horizontal"),
    ]
)


def graph_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


# Dataset & splits
train_ds = tfds.load(DS_NAME, split='train', as_supervised=True).map(
    preprocess_data).map(lambda x, y: (data_augmentation(x), y)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

validation_ds = tfds.load(DS_NAME, split='validation', as_supervised=True).map(
    preprocess_data).batch(VALIATION_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Our classification model
if LOAD_MODEL:
    model = keras.models.load_model(MODEL_FILENAME)
else:
    model = keras_cv.models.ImageClassifier.from_preset(
        "efficientnetv2_b0_imagenet", num_classes=len(CLASSES)
    )

if TRAIN_MODEL:
    # Phase A
    model.get_layer('efficient_net_v2b0_backbone').trainable = False
    model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS_A,
    )

    graph_history(history)

    # Phase B
    model.get_layer('efficient_net_v2b0_backbone').trainable = True
    model.summary()

    history2 = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS_B,
    )

    graph_history(history2)

if SAVE_MODEL:
    model.save(MODEL_FILENAME)

#  Show validation pictures along with their original & inferred categories
if SHOW_PREDS:
    results = []

    # Accumulate all predictions
    for tensors in validation_ds:
        images, labels = tensors[0], tensors[1]
        predictions = model.predict(images)

        for idx, image in enumerate(images):
            results.append({
                'image': image,
                'real_class': CLASSES[tf.math.argmax(labels[idx]).numpy()],
                'pred_class': CLASSES[tf.math.argmax(predictions[idx]).numpy()]
            })

    # Show them
    ic_fig = plt.figure(num=1, figsize=(8, 8))
    ic_count = 0

    for entry in results:
        sp = ic_fig.add_subplot(2, 2, ic_count + 1)
        sp.set_title(entry['real_class'] + ' / ' + entry['pred_class'], size=8)
        plt.imshow(keras.utils.array_to_img(entry['image']))
        ic_count += 1

        if ic_count == 4:
            plt.show()
            ic_fig = plt.figure(num=1, figsize=(8, 8))
            ic_count = 0

    if ic_count > 0:
        plt.show()

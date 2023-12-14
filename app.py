import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Function to create and compile the model
def create_model():
    NUM_CLASSES = 5  # We hebben natuurlijk 5 classes
    IMG_SIZE = 128  # De foto's zijn 128 op 128 pixels
    HEIGHT_FACTOR = 0.2  # Maximale afwijking van de data augmentatie
    WIDTH_FACTOR = 0.2
    model = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(HEIGHT_FACTOR, WIDTH_FACTOR),
        layers.RandomZoom(0.2),
        layers.Conv2D(48, (5, 5), input_shape=(96, 96, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Function to train the model
def train_model(model, training_set, validation_set, epochs):
    history = model.fit(training_set, validation_data=validation_set, epochs=epochs)
    return history

# Function to display training and validation curves
def display_training_curves(history):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    fig.tight_layout()
    st.pyplot(fig)

# Function to evaluate the model using a confusion matrix
def evaluate_model(model, test_set, categories):
    true_labels = []
    predicted_labels = []
    steps = len(test_set)

    for i in range(steps):
        #De namen moeten ook x_batch en y_batch zijn of het werkt niet
        x_batch, y_batch = test_set[i]
        true_labels.extend(np.argmax(y_batch, axis=1))
        predicted_labels.extend(np.argmax(model.predict(x_batch), axis=1))

    cm = confusion_matrix(true_labels, predicted_labels)

    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    plot.plot(cmap='Greens')
    plt.title('Confusion Matrix')
    st.pyplot()

    # Make predictions on a single image
    img_path = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])
    if img_path is not None:
        predict_single_image(model, img_path, categories)

# Function to make predictions on a single image
def predict_single_image(model, img_path, categories):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_category = categories[np.argmax(predictions)]

    st.image(img, caption=f"Predicted category: {predicted_category}")

# Streamlit app
def main():
    st.title("Monument Classifier App")

    # Load and preprocess data
    # ...
    categories = ['Atomium', 'Coloseum', 'Eiffel Tower', 'Statue of Liberty', 'Sydney Opera House']

    # Add a button to trigger model training
    train_button = st.button("Train Model")

    if train_button:
        train_val_datagen = ImageDataGenerator(
            validation_split=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_val_datagen.flow_from_directory(
            'datasets/training_set',
            subset='training',
            target_size=(96, 96),
            batch_size=32,
            class_mode='categorical'
        )

        validation_set = train_val_datagen.flow_from_directory(
            'datasets/training_set',
            subset='validation',
            target_size=(96, 96),
            batch_size=32,
            class_mode='categorical'
        )

        test_set = test_datagen.flow_from_directory(
            'datasets/testing_set',
            target_size=(96, 96),
            batch_size=32,
            class_mode='categorical'
        )

        # Create and train the model
        model = create_model()
        history = train_model(model, training_set, validation_set, epochs=10)

        # Display training curves
        display_training_curves(history)

        # Evaluate the model using a confusion matrix
        evaluate_model(model, test_set, categories)

    

if __name__ == "__main__":
    main()

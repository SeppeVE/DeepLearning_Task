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


# Function to load the model
@st.cache_resource
def load_custom_model():
    # Load the pre-trained model
    model = keras.models.load_model("saved_models/monuments.tf")
    evaluate_model()
    return model


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

    model = None

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
        model = load_custom_model()

        # Evaluate the model using a confusion matrix
        evaluate_model(model, test_set, categories)

    # Make predictions on a single image
    img_path = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])
    if img_path is not None:
        predict_single_image(model, img_path, categories)

if __name__ == "__main__":
    main()

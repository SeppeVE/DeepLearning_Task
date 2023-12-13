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
    model = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(HEIGHT_FACTOR, WIDTH_FACTOR),
        layers.RandomZoom(0.2),
        layers.Conv2D(64, (5, 5), input_shape=(128, 128, 3), activation="relu"),
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

    for i, (x_batch, y_batch) in enumerate(test_set):
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

    train_val_datagen = ImageDataGenerator(
            validation_split=0.2,     # De training data wordt voor 80% gebruikt om te trainen en 20% om te valideren.
            rescale=1./255,           # De pixels van de foto's worden omgezet van een waarde tussen 0 en 255 naar een waarde tussen 0 en 1. Dit is omdat de neuronen die de pixel waarden bevatten enkel getallen tussen 0 en 1 kunnen gebruiken.
            shear_range=0.2,          # Shear wordt toegepast met een afwijking tussen de -20% en 20% van de originele afbeelding.
            zoom_range=0.2,           # Ook hier geldt het -20% tot 20% principe maar dan op het in- of uitzoomen van de afbeelding.
            horizontal_flip=True,     # De foto's worden ook nog eens horizontaal geflipt.
        )

    test_datagen = ImageDataGenerator(rescale = 1./255)

    # Training data word gemaakt uit de training_set directory. Hier wordt dan de eerder ingestelde data augmentatie op worden toegepast.
    training_set = train_val_datagen.flow_from_directory(
        'datasets/training_set',     # Verwijzing naar de folder met de data.
        subset='training',       # We willen hier de training subset gebruiken. Dit is de 90% die eerder gesplit werd.
        target_size=(128, 128),    # De foto's worden geresized naar 128 bij 128 pixels
        batch_size=32,           # Per Epoch gaan we om de 32 foto's even evalueren en de nodige parameters aanpassen.
        class_mode='categorical' # We willen uiteindelijk foto's die binnen een bepaalde categorie vallen. Dus moet hier niet binary zoals met 2 outputs maar categorical voor 5 outputs.
    )

    # De validatie data word aangemaakt. De instellingen zijn verder zoals bij de training data.
    validation_set = train_val_datagen.flow_from_directory(
        'datasets/training_set',     
        subset='validation',     
        target_size=(128, 128),    
        batch_size=32,           
        class_mode='categorical'
    )

    # De test data word aangemaakt. De instellingen zijn verder zoals bij de training data.
    test_set = test_datagen.flow_from_directory(
        'datasets/testing_set',      
        target_size=(128, 128),    
        batch_size=32,           
        class_mode='categorical' 
    )
    # Create and train the model
    model = create_model()
    history = train_model(model, training_set, validation_set, epochs=45)

    # Display training curves
    display_training_curves(history)

    # Evaluate the model using a confusion matrix
    evaluate_model(model, test_set, categories)

    # Make predictions on a single image
    img_path = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])
    if img_path is not None:
        predict_single_image(model, img_path, categories)

if __name__ == "__main__":
    main()

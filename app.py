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

st.set_option('deprecation.showPyplotGlobalUse', False)
# Function to load the model
@st.cache_resource
def load_custom_model():
    # Load the pre-trained model
    model = keras.models.load_model("saved_models/monuments.tf")
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

def main():
    st.title("Monument Classifier App")
    st.text("In deze applicatie zult u er voor kunnen kiezen om een AI model te kunnen testen. Waarop? Verschillende monumenten van over de wereld. Het model is getrained om te zeggen naar welk monument het kijkt. Natuurlijk kunnen niet alle monumenten er inzitten dus hebt u de keuze tussen 5 opties.")
    st.text("Atomium")
    st.text("Colosseum")
    st.text("Eiffeltoren")
    st.text("Statue of Liberty")
    st.text("Sydney Opera House")
    st.text("Volg de onderstaande stappen om de AI uit te dagen.")
    categories = ['Atomium', 'Colosseum', 'Eiffel Tower', 'Statue of Liberty', 'Sydney Opera House']

    show_training_results = st.checkbox("Show Training Results")
    model = None

    if show_training_results:

        test_datagen = ImageDataGenerator(rescale=1./255)

        test_set = test_datagen.flow_from_directory(
            'datasets/testing_set',
            target_size=(96, 96),
            batch_size=32,
            class_mode='categorical'
        )

        # Create and train the model only if the checkbox is selected
        model = load_custom_model()
        evaluate_model(model, test_set, categories)

    # Check if the model is loaded before allowing image upload
    if model is not None:
        # Make predictions on a single image
        img_path = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])
        if img_path is not None:
            predict_single_image(model, img_path, categories)
    else:
        st.warning("Als u de checkbox aanvinkt zal het model geladen worden. Hierna zult u de optie krijgen om een foto van een van de monumenten te uploaden waarop een output geprint zal worden met de foto en wat het model denkt dat het monument is.")

if __name__ == "__main__":
    main()

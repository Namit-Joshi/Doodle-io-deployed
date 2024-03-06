from flask import Flask, request, render_template
from flask_cors import cross_origin
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import numpy as np
import cv2

# Initialize the useless part of the base64 encoded image.
init_Base64 = 21

# Dictionary mapping class indices to class labels
label_dict = {0: 'Sheep', 1: 'Butterfly', 2: 'Octopus', 3: 'Hedgehog', 4: 'Duck', 5: 'Fish'}

app = Flask(__name__, template_folder='templates')

# Eager execution is enabled by default in TensorFlow 2.x, but let's make sure
tf.config.run_functions_eagerly(True)

@app.route("/")
@app.route("/home")
@cross_origin()
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        model = load_model("quickdraw.h5")
        final_pred = None

        # Preprocess the image: set the image to 28x28 shape
        # Access the image
        draw = request.form['url']

        # Removing the useless part of the url.
        draw = draw[init_Base64:]

        # Decoding
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        # Resizing and reshaping to keep the ratio.
        resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        vect = np.asarray(resized, dtype="uint8")

        vect = vect.reshape(1, 28, 28, 1).astype('float32')
        vect = vect / 255

        # Launch prediction
        my_prediction = np.argmax(model.predict(vect), axis=-1)

        # Getting the index of the maximum prediction
        final_pred = label_dict[my_prediction[0]]

    return render_template('results.html', prediction=final_pred)

if __name__ == "__main__":
    app.run(debug=True)

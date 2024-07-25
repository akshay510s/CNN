from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load the model
model = load_model('cnn_model.h5')

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    prediction = model.predict(np.expand_dims(img, axis=0))
    return jsonify({'expression': CLASSES[np.argmax(prediction)]})

if __name__ == '__main__':
    app.run(port=5000)

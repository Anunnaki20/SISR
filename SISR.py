# Web Framework
from flask import Flask
from flask import request
from flask import jsonify

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

# print('TensorFlow version: {}'.format(tf.__version__))
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Create the Flask Web application
app = Flask(__name__)

# Base URL
@app.route('/', methods=["POST"])
def test():
    data = request.get_json()
    print(data)
    return jsonify(f"Hey! {data}")


# Run the server on the local host
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
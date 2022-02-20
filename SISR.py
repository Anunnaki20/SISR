# Web Framework
from flask import Flask
from flask import request, redirect, Response
from flask import jsonify

import json
import cv2
import os
import io
import PIL.Image as Image
import base64

# Tensorflow libraries
# import tensorflow as tf
# from tensorflow import keras

# Helper libraries
import numpy as np
# import matplotlib.pyplot as plt
# import os
# import subprocess

# print('TensorFlow version: {}'.format(tf.__version__))
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Create the Flask Web application
app = Flask(__name__)

# Base URL
@app.route('/', methods=['GET', 'POST'])
def test():
    # handle the POST request
    if request.method == 'POST':

        r = request
        # info = request.get_json()
        # print(jsonify(info.model))
        # print(json.loads(info)["model"])
        # print(r.json()["model"])
        # print(r.data["model"])
        # print(r.data["scaleAmount"])
        # data = json.loads(r.data)
        # print(r.json)
        # stuff = json.loads(r.text)
        # print(stuff["scaleAmount"])

        # print(str(r.data))

        scale = r.json['scaleAmount']
        model = r.json['model']
        print("Scale:", scale, ", Model:", model)
        # data = r.data['image']

        # nparr = np.frombuffer(arr, np.uint8)
        # img = base64.b64decode(arr)
        # nparr = np.frombuffer(img, np.uint8)
        # nparr = np.asarray(arr)
        
        # data = base64.b64decode(arr)
        # print(str(scale), str(model))
        # print(data["model"])
        # print()
        # print(data["scaleAmount"])
        nparr = np.frombuffer(r.data, np.uint8)

        # convert string of image data to uint8
        # nparr = np.frombuffer(r.data, np.uint8)
        # print(r.data)
        # print()
        # print(r.headers)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) #IMREAD_GRAYSCALE #IMREAD_COLOR
        # Now we can do whatever we want with this decoded image ...

        # new_img = cv2.imread("testimage.png", cv2.IMREAD_GRAYSCALE)
        # # Displaying the image
        # cv2.imshow('image', arr)

        ############## Save the image as a png ##############
        # cv2.imwrite("testimage.png", img)
        

        # This code here is to send back to the "client" the front-end to show that it received the image and reads it properly.
        # build a response dict to send back to client
        response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                    }
        print(response)
        # encode response using jsonpickle
        # response_pickled = jsonpickle.encode(response)

        # return Response(response=response_pickled, status=200, mimetype="application/json")
        data = "Nothing"


        ######################################################################################################################################################
        ######################################################################################################################################################
        ############### The following code was used for testing. Keep this here just in case we need to send the image or files in this way ##################
        ######################################################################################################################################################
        ######################################################################################################################################################
        # print(request.files)
        # data = request.get_data()
        # print(data)
        # print(request.files)
        # if 'file' not in request.files:
        #     print('No file part')
        #     return redirect(request.url)
        # file = request.files['file']
        # if file.filename == '':
        #     print('No image selected for uploading')
        #     return redirect(request.url)
        # # if file and allowed_file(file.filename):
        # else:
        #     filename = file.filename
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #     #print('upload_image filename: ' + filename)
        #     print('Image successfully uploaded and displayed below')
        #     # return redirect(request.url)
        #     # return render_template('upload.html', filename=filename)

        
        # data = request.get_json()
        # data = request.get_data()
        # if data is None:
        #     print("dooodoo")
        # print(data)
        # print(request.files)
        # #read image file string data
        # filestr = request.files['file'].read()
        ######################################################################################################################################################
        ######################################################################################################################################################
        ######################################################################################################################################################

        return jsonify(f"Hey! {data}")

    # otherwise handle the GET request
    else:
        return jsonify(f"Hey!")



# Run the server on the local host
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
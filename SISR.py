# Web Framework
from flask import Flask
from flask import request, redirect, Response
from flask import jsonify


import cv2
import os
import io
import PIL.Image as Image

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
        # convert string of image data to uint8
        nparr = np.frombuffer(r.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) #IMREAD_GRAYSCALE #IMREAD_COLOR
        # Now we can do whatever we want with this decoded image ...

        # new_img = cv2.imread("testimage.png", cv2.IMREAD_GRAYSCALE)
        # # Displaying the image
        # cv2.imshow('image', img)

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
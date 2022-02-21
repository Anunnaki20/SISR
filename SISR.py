# Web Framework
from email.mime import base
from flask import Flask
from flask import request, redirect, Response
from flask import jsonify


import cv2
import os
import PIL.Image as Image
import zipfile
import shutil

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

        parameters = r.args
        # print(r.args)
        filetype = parameters['type']
        scale = parameters['scaleAmount']
        model = parameters['model']
        qualityMeasure = parameters['qualityMeasure']
        print("File Type:", filetype, ", Scale:", scale, ", Model:", model, ", Quality Measure?:", qualityMeasure)

        if filetype == "zip":
            zip_result = open('./uploadedFile/testzip.zip', 'wb')
            zip_result.write(r.data)

            ######################################################
            # unzip the file and check image size for each image #
            ######################################################
            file_url = "/uploadedFile/testzip.zip"
            # extract the images from the zip
            with zipfile.ZipFile("."+file_url, 'r') as zip_ref:
                zip_ref.extractall("./uploadedFile/extractedImages")

            ####################################
            # Upscale each image in the folder #
            ####################################


        else: #filetype == "single_image"
            # convert string of image data to uint8
            nparr = np.frombuffer(r.data, np.uint8)
            # decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) #IMREAD_GRAYSCALE #IMREAD_COLOR
            
            response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                    }
            print(response)

            ############## Save the image as a png ##############
            cv2.imwrite("./uploadedFile/decodedimage.png", img)

            ###################################
            # Upscale the image in the folder #
            ###################################


        ########################
        # Zip the single image #
        ########################
        shutil.make_archive("./upscaledImages/upscaled", 'zip', "./uploadedFile/extractedImages")
        file_url = "/upscaledImages/upscaled.zip"

        ################################
        # Send the zip back to website #
        ################################

        ##########################
        # Delete all saved files #
        ##########################
        cleanDirectories()

        # This is just a dummy variable
        data = "Nothing"

        ######################################################################################################################################################
        ######################################################################################################################################################
        ######################################################################################################################################################

        return jsonify(f"Hey! {data}")

    # otherwise handle the GET request
    else:
        return jsonify(f"Hey!")

# Remove/delete the files in the images and extractedImages folders
def cleanDirectories():
    ####################################
    # Delete the items in subdirectory #
    ####################################
    for file_in_sub in os.listdir("./uploadedFile/extractedImages"):
        if os.path.isdir("./uploadedFile/extractedImages/"+file_in_sub):
            try:
                shutil.rmtree("./uploadedFile/extractedImages/"+file_in_sub)
                # os.rmdir("./images/extractedImages/"+file_in_sub)
            except OSError as e:
                print("Error: %s : %s" % ("./uploadedFile/extractedImages/"+file_in_sub, e.strerror))
        else:
            try:
                os.remove("./uploadedFile/extractedImages/"+file_in_sub)
            except OSError as e:
                print("Error: %s : %s" % ("./uploadedFile/extractedImages/"+file_in_sub, e.strerror))

    ##############################################
    # Delete the items in uploadedFile directory #
    ##############################################
    for file_in_main in os.listdir("./uploadedFile"):
        if os.path.isdir("./uploadedFile/"+file_in_main): # item is a directory
            continue # do not delete
        elif os.path.isfile("./uploadedFile/"+file_in_main): # item is a file
            try:
                os.remove("./uploadedFile/"+file_in_main)
            except OSError as e:
                print("Error: %s : %s" % ("./uploadedFile/"+file_in_main, e.strerror))

    ################################################
    # Delete the items in upscaledImages directory #
    ################################################
    for file_in_upscaled in os.listdir("./upscaledImages"):
        if os.path.isdir("./upscaledImages/"+file_in_upscaled): # item is a directory
            continue # do not delete
        elif os.path.isfile("./upscaledImages/"+file_in_upscaled): # item is a file
            try:
                os.remove("./upscaledImages/"+file_in_upscaled)
            except OSError as e:
                print("Error: %s : %s" % ("./upscaledImages/"+file_in_upscaled, e.strerror))

# Run the server on the local host
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
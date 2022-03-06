# Web Framework
from email.mime import base
import io
import itertools
from pathlib import Path
from flask import Flask
from flask import request, redirect, Response
from flask import jsonify
from flask import send_file

# import tensorflow as tf
import time
import os
# import PIL.Image as Image
import zipfile
import shutil
import base64
import skimage.io
from multiprocessing.dummy import Pool as ThreadPool

# Machine learning libraries
from keras.models import load_model

# import sisrPredict

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import sys
import numpy as np
# import matplotlib.pyplot as plt
# import subprocess

# print('TensorFlow version: {}'.format(tf.__version__))
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'
#session = tf.compat.v1.Session(config=config)
#with tf.compat.v1.Session(config = config) as s:

# Create the Flask Web application
app = Flask(__name__)


# Model Uploading URL
@app.route('/uploadModel', methods=['POST'])
def modelUploading():

    # handle the POST request
    if request.method == 'POST':

        r = request

        parameters = r.args
        modelDesc = parameters['modelDesc']
        filename = parameters['filename']

        ############################
        # store the uploaded model #
        ############################
        modelPath = "./models/" + filename

        print("filename: " + filename + " , modelDesc: " + modelDesc)
        # filedata = base64.b64decode(r.data)
        filedata = r.data

        with open(modelPath, 'wb') as f:
            f.write(filedata)
            
        # This is just a dummy variable
        data = "Nothing"

        ######################################################################################################################################################
        ######################################################################################################################################################
        ######################################################################################################################################################
       
        # Delete all saved files #
        ##########################
        # cleanDirectories()
        
        return jsonify(f"Hey! {data}")
        #return filetype, scale, model, qualityMeasure, img

    # otherwise handle the GET request
    else:
        return jsonify(f"Hey!")


# Base URL
@app.route('/', methods=['GET', 'POST'])
def test():

    # handle the POST request
    if request.method == 'POST':

        r = request

        parameters = r.args
        filetype = parameters['type']
        scale = parameters['scaleAmount']
        modelName = "models/" + parameters['model']
        qualityMeasure = parameters['qualityMeasure']

        zippedFiles = []

        if filetype == "zip":

            print("File Type:", filetype, ", Scale:",  scale, ", Model:", modelName, ", Quality Measure?:", qualityMeasure)  
            
            ######################################################
            # store the zipped folder #
            ######################################################
            zipPath = "./uploadedFile/uploaded.zip"
            with open(zipPath, 'wb') as zipFile:
                zipFile.write(r.data)
            
            # extract the images from the zip
            with zipfile.ZipFile(zipPath, 'r') as zip_ref:
                zippedFiles = zip_ref.namelist()
                zip_ref.extractall("./uploadedFile/extractedImages")


            zippedFilesPath = [ "./uploadedFile/extractedImages/" + s for s in zippedFiles]
            # Load CNN
            startTimeX = time.time() 

            ####################################
            # Upscale each image in the folder #
            ####################################  
            gtImageFiles = [skimage.io.imread(im) for im in zippedFilesPath]
            total_image = len(gtImageFiles)

            # Multithreading = (took me 25s for 8 files, 68s for 32 files, each x4 upscale)
            if total_image>1:
                pool = ThreadPool(os.cpu_count())
                pool.starmap(upScaleImage,zip(itertools.repeat(modelName),zippedFiles,gtImageFiles,itertools.repeat(qualityMeasure),itertools.repeat(int(scale)),itertools.repeat(total_image)))
            else:
                upScaleImage(modelName, zippedFiles[0], gtImageFiles[0], qualityMeasure, int(scale), total_image)
            # for loop = (took me 50s-60s for 8 files, 192s for 32 files, each x4 upscale)
            # zippedFiles = [ "./uploadedFile/extractedImages/" + s for s in zippedFiles]
            # for file in zippedFiles:
            #     basename = Path(file).stem
            #     gtImage = skimage.io.imread(file)
            #     upScaleImage(modelName, basename, gtImage, qualityMeasure, int(scale))
                
            print("Time to finish upscaling = %f" % (time.time()  - startTimeX)) 

            ##################
            # Zip the images #
            ##################
            shutil.make_archive("./upscaledImages/upscaled", 'zip', "./uploadedFile/extractedImages")
            file_url = "/upscaledImages/upscaled.zip"


        else: #filetype == "single_image"
            # payload = {'single': True}
            # convert string of image data to uint8
            fileName = parameters['filename']
            print("File Type:", filetype, ", Scale:",  scale, "filename ", fileName, ", Model:", modelName, ", Quality Measure?:", qualityMeasure)            
            imgdata = base64.b64decode(r.data)
            img = Image.open(io.BytesIO(imgdata))
            img = np.asarray(img)
            # decode image
            #img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) #IMREAD_GRAYSCALE #IMREAD_COLOR
            
            #response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}

            ############## Save the image as a png ##############
            #cv2.imwrite("./uploadedFile/decodedimage.png", img)

            ###################################
            # Upscale the image in the folder #
            ###################################
            # Load CNN
            startTimeX = time.time()
            
            # Upscale the image
            upScaleImage(modelName, fileName, img, qualityMeasure, int(scale))
            print("Total time to upscale = %f" % (time.time()  - startTimeX))

        shutil.make_archive("./upscaledZip", 'zip', './upscaledImages')
        ########################
        # Zip the single image #
        ########################
        
        #file_url = "/upscaledImages/upscaled.zip"

        ################################
        # Send the zip back to website #
        ################################
        ##########################


        # This is just a dummy variable
        data = "Nothing"

        ######################################################################################################################################################
        ######################################################################################################################################################
        ######################################################################################################################################################
    
        # Delete all saved files #
        ##########################
        # cleanDirectories()
        
        return jsonify(f"Hey! {data}")
        #return filetype, scale, model, qualityMeasure, img

    # otherwise handle the GET request
    else:
        return jsonify(f"Hey!")


def upScaleImage(modelName, filename, img, qualityMeasure, scale, total_image):
    # Load CNN
    model = load_model(modelName, compile=False)
    #model.summary()
    # Upscale the image
    sisrPredict.predict(model, filename, img, qualityMeasure, scale, total_image)   

# Send upscaled image to download
# @app.route('/downloadImage')
# def sendImage():
#     for file in os.listdir("./upscaledImages"):
#         if file.endswith(".png"):
#             try:
#                 return send_file('./upscaledImages/'+file, as_attachment=True)
#             except Exception as e:
#                 return str(e)

# Send upscaled zip folder to download
@app.route('/downloadZip', methods=['GET','POST'])
def sendZip():
    
    ### Check the already created upscaled zip folder and then create a new zipFile object on memory ###
    FILEPATH = "./upscaledZip.zip"
    fileobj = io.BytesIO()
    with zipfile.ZipFile(fileobj, 'w') as zip_file:
        zip_info = zipfile.ZipInfo(FILEPATH)
        zip_info.date_time = time.localtime(time.time())[:6]
        zip_info.compress_type = zipfile.ZIP_DEFLATED
        with open(FILEPATH, 'rb') as fd:
            zip_file.writestr(zip_info, fd.read())
    fileobj.seek(0)

    """response = make_response(fileobj.read())
    response.headers.set('Content-Type', 'zip')
    response.headers.set('Content-Disposition', 'attachment', filename='upscaledZip.zip')
    return response""" # This allows zip to be downloaded in user machine

    with open("./upscaledZip.zip", 'rb') as f:
        data = f.readlines()
    
    cleanDirectories()
    
    return Response(data, 
        headers={
        'Content-Type': 'application/zip',
        'Content-Disposition':'attachment;filename=upscaledZip.zip'
    })
    #return send_file(fileobj.read(), mimetype='zip', as_attachment=True, attachment_filename = '%s' % os.path.basename(FILEPATH))




#@app.route("/")
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

    if os.path.exists("./upscaledZip.zip"):
        os.remove("./upscaledZip.zip")

# Run the server on the local host
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
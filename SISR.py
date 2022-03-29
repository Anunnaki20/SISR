# Web Framework
import io
import itertools
from flask import Flask
from flask import request, Response
from flask import jsonify

import time
import os
import PIL.Image as Image
import zipfile
import shutil
import base64
import skimage.io
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

# Machine learning libraries
from keras.models import load_model
import sisrPredict

# Helper libraries
import numpy as np

import csv
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
        filedata = r.data

        with open(modelPath, 'wb') as f:
            f.write(filedata)
        
        return ("", 204)


    # otherwise handle the GET request
    else:
        return ("", 204)


# Base URL
@app.route('/', methods=['POST'])
def upload():

    cleanDirectories()  # Clean all previously saved directories
    
    # handle the POST request
    if request.method == 'POST':

        r = request

        parameters = r.args
        filetype = parameters['type']
        scale = parameters['scaleAmount']
        modelName = "models/" + parameters['model']
        qualityMeasure = parameters['qualityMeasure']

        zippedFiles = []

        # Initialize all quality metrics
        nnMSE = blMSE = bcMSE = rcMSE = diffMSE = nnPSNR = blPSNR = bcPSNR = rcPSNR = diffPSNR = nnSSIM = blSSIM = bcSSIM = rcSSIM = diffSSIM = 0
        
        # Create the comparison result file
        comparison = open("upscaledImages/comparisonResult.txt", "a")

        if filetype == "zip":

            print("File Type:", filetype, ", Scale:",  scale, ", Model:", modelName, ", Quality Measure?:", qualityMeasure)  
            
            ######################################################
            # store the zipped folder #
            ######################################################
            if not os.path.isdir("uploadedFile"):
                os.mkdir("uploadedFile")
            zipPath = "./uploadedFile/uploaded.zip"
            with open(zipPath, 'wb') as zipFile:
                zipFile.write(r.data)
            
            # extract the images from the zip
            with zipfile.ZipFile(zipPath, 'r') as zip_ref:
                zippedFiles = zip_ref.namelist()
                zip_ref.extractall("./uploadedFile/extractedImages")


            zippedFilesPath = [ "./uploadedFile/extractedImages/" + s for s in zippedFiles]
            # Load CNN
            

            ####################################
            # Upscale each image in the folder #
            ####################################  
            gtImageFiles = [skimage.io.imread(im) for im in zippedFilesPath]
            total_image = len(gtImageFiles)


            # Multithreading = (took me 25s for 8 files, 68s for 32 files, each x4 upscale)
            pool = ThreadPool(os.cpu_count())

            startTimeX = time.time() 
            # print(type(qualityMeasure))
            if qualityMeasure=="True" and total_image>0:
                results = pool.starmap(upScaleImage,zip(itertools.repeat(modelName),zippedFiles,gtImageFiles,itertools.repeat(qualityMeasure),itertools.repeat(int(scale)),itertools.repeat(total_image)))
                #print(results)
                for index, result in enumerate(results):
                    nnMSE += result[0]
                    blMSE += result[1]
                    bcMSE += result[2]
                    rcMSE += result[3]
                    diffMSE += result[4]
                    nnPSNR += result[5]
                    blPSNR += result[6]
                    bcPSNR += result[7]
                    rcPSNR += result[8]
                    diffPSNR += result[9]
                    nnSSIM += result[10]
                    blSSIM += result[11]
                    bcSSIM += result[12]
                    rcSSIM += result[13]
                    diffSSIM += result[14]

                comparison.write('%12.6f,%12.6f,%12.6f,%12.6f,%12.6f\n' % 
                (nnMSE/total_image, blMSE/total_image, bcMSE/total_image, rcMSE/total_image, diffMSE/total_image))
                comparison.write('%12.6f,%12.6f,%12.6f,%12.6f,%12.6f\n' % 
                (nnPSNR/total_image, blPSNR/total_image, bcPSNR/total_image, rcPSNR/total_image, diffPSNR/total_image))
                comparison.write('%12.6f,%12.6f,%12.6f,%12.6f,%12.6f\n' % 
                (nnSSIM/total_image, blSSIM/total_image, bcSSIM/total_image, rcSSIM/total_image, diffSSIM/total_image))

                comparison.close()

                # Write to the csv file
                read_file = pd.read_csv ('upscaledImages/comparisonResult.txt', sep=",",names=["NearNeighbour", "Bi-linear", "Bi-cubic", "Reconstruct","Difference"])
                read_file.index=["MSE", "PSNR", "SSIM"]
                read_file.to_csv ('upscaledImages/comparisonResult.csv',sep=',')
                
            
            else:
                pool.starmap(upScaleImage,zip(itertools.repeat(modelName),zippedFiles,gtImageFiles,itertools.repeat(qualityMeasure),itertools.repeat(int(scale)),itertools.repeat(total_image)))

            print("Time to finish upscaling = %f" % (time.time()  - startTimeX)) 

        else: #filetype == "single_image"
            fileName = parameters['filename']
            print("File Type:", filetype, ", Scale:",  scale, "filename ", fileName, ", Model:", modelName, ", Quality Measure?:", qualityMeasure)            
            imgdata = base64.b64decode(r.data)
            img = Image.open(io.BytesIO(imgdata))
            img = np.asarray(img)

            ###################################
            # Upscale the image in the folder #
            ###################################
            # Load CNN
            startTimeX = time.time()
            
            # Upscale the image
            if qualityMeasure=="True":
                result = upScaleImage(modelName, fileName, img, qualityMeasure, int(scale), 1)
                comparison.write('%12.6f,%12.6f,%12.6f,%12.6f,%12.6f\n' % 
                (result[0], result[1], result[2], result[3], result[4]))
                comparison.write('%12.6f,%12.6f,%12.6f,%12.6f,%12.6f\n' % 
                (result[5], result[6], result[7], result[8], result[9]))
                comparison.write('%12.6f,%12.6f,%12.6f,%12.6f,%12.6f\n' % 
                (result[10], result[11], result[12], result[13], result[14])) 

                comparison.close()

                # Write to the csv file
                read_file = pd.read_csv ('upscaledImages/comparisonResult.txt', sep=",",names=["NearNeighbour", "Bi-linear", "Bi-cubic", "Reconstruct","Difference"])
                read_file.index=["MSE", "PSNR", "SSIM"]
                read_file.to_csv ('upscaledImages/comparisonResult.csv',sep=',')

            else:
                upScaleImage(modelName, fileName, img, qualityMeasure, int(scale), 1)

            print("Total time to upscale = %f" % (time.time()  - startTimeX))

        # Zips upscaled images
        shutil.make_archive("./upscaledZip", 'zip', './upscaledImages')

        return ("", 204)

    # otherwise handle the GET request
    else:
        return ("", 204)


# Call the CNN
def upScaleImage(modelName, filename, img, qualityMeasure, scale, total_image):
    # Load CNN
    model = load_model(modelName, compile=False)
    # Upscale the image
    return sisrPredict.predict(model, filename, img, qualityMeasure, scale, total_image)   


# Send upscaled zip folder to download
@app.route('/downloadZip', methods=['GET','POST'])
def sendZip():
    
    try:
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

        with open("./upscaledZip.zip", 'rb') as f:
            data = f.readlines()
        
        if os.path.isdir("./upscaledZip.zip"):
            os.remove("./upscaledZip.zip")
        cleanDirectories()
        
        return Response(data, 
            headers={
            'Content-Type': 'application/zip',
            'Content-Disposition':'attachment;filename=upscaledZip.zip'
        })

    except Exception as e:
        print("Error in sendZip")
        cleanDirectories()
        return Response(None,status=500)




# Remove/delete the files in the images and extractedImages folders
def cleanDirectories():
    ####################################
    # Delete the items in subdirectory #
    ####################################
    if os.path.exists('./uploadedFile/extractedImages/'):
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
    if os.path.exists('./uploadedFile'):
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
    if os.path.exists('./upscaledImages'):
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
    app.run(debug=False, host='0.0.0.0')
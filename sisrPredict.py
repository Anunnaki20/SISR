#************************************************************
# CMPT819 Project SISR - Predict
# Andrew Kostiuk, andrew.kostiuk@usask.ca
# December 3, 2017
# Copyright (c) 2017, Andrew Kostiuk
#------------------------------------------------------------

# All the imports we need
from operator import gt
import re
import warnings
import sys
import time
import numpy
import os
import datetime
import multiprocessing as mp

# Machine learning libraries
import keras
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.client import device_lib

# image manipulation libraries

# Original
import skimage.io
import skimage.transform
import skimage.color
import cv2

# keras.backend.set_learning_phase(0)

#------------------------------------------------------------
# Settings
#------------------------------------------------------------

# Turn off warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

inPatchRows = 128
inPatchCols = 128
modelDir = 'models'
outputDir = 'output'

n1 = 64
n2 = 32
f1 = 9
f2 = 1
f3 = 5
offset = int(f1/2) + int(f3/2)

trainingPatches = 1024

delimiterMajor = '============================================================'
delimiter = '----------------------------------------------------------'

os.environ["CUDA_VISIBLE_DEVICES"]="0"   

# print(tf.__version__)
# print('A: ', tf.test.is_built_with_cuda)
# print('B: ', tf.test.gpu_device_name())
local_device_protos = device_lib.list_local_devices()
([x.name for x in local_device_protos if x.device_type == 'GPU'], 
 [x.name for x in local_device_protos if x.device_type == 'CPU'])

# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())

# Make directories if necessary
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

# Open the loggin file
log = open(outputDir + '/sisrPredict.log', 'a+')
#------------------------------------------------------------
# Used to time fucntions
def _time(f):
    def wrapper(*args):
        start = time.time()
        r = f(*args)
        end = time.time()
        print("%s timed %f" % (f.__name__, end-start) )
        return r
    return wrapper


# Logging 
def xprint(string):
    """Writes string data to the log file with the current time and date.

    Args:
        string (string): The string value you want to right to the log file
    """
    string = (datetime.datetime.now().strftime("%H:%M:%S")) + ' ' + string
    print(string)
    log.write(string + '\n')
    log.flush()
    return

# Get input 
@_time
def Load_inputs():
    """Reads the system args to determine the input file, scale amount, and either upscale or downscale

    Returns:
        string, string, bool, int: the name of the CNN model, name of the image to upscale, if we are downscale, scale amount
    """

    # If not enough arguments are given. Print what it should be then quit
    if len(sys.argv) < 4:
        xprint('Usage: python sisrPredict.py <input_image> -s <2|4> [-d]')
        xprint(delimiterMajor)
        quit()
        
    # if the file is not found quit
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        xprint('Input file "%s" not found.' % (filename))
        xprint(delimiterMajor)
        quit()

    # If the second argument is not -s quit
    if sys.argv[2] != '-s':
        xprint('Second parameter "%s" not scale.' % (sys.argv[2]))
        xprint(delimiterMajor)
        quit()

    # The scale must be x2 or x4 
    scale = int(sys.argv[3])
    if scale != 2 and scale != 4:
        xprint('Invalid scale = %d.' % (scale))
        xprint(delimiterMajor)
        quit()

    # If they have 4 args then check that its -d. If it is set downsample to true else quit
    downsample = False
    if len(sys.argv) > 4:
        if sys.argv[4] != '-d':
            xprint('Invalid parameter "s".' % (sys.argv[4]))
            xprint(delimiterMajor)
            quit()
        else:
            downsample = True

    # log the settings       
    xprint('Parameters:')
    xprint('    scale = %d' % (scale))
    xprint('    trainingPatches = %d' % (trainingPatches))

    xprint('    downsample = %s' % (str(downsample)) )
    xprint(delimiter)

    # Load model
    modelName = '%s/model_%d_t%03d_e1000.h5' % (modelDir, scale, trainingPatches)
    if not os.path.isfile(modelName):
        xprint('Model "%s" not found.' % (modelName))
        xprint(delimiterMajor)
        quit()

    return modelName, filename, downsample, scale
    

# Helper functions
@_time
def DetermineComparisons(image1, image2):
    
    """Compares 2 images together using Mean squared error, Peak signal to noise ratio, and structural similarity

    Args:
        image1 (numpy array): The first image we are comparing with
        image2 (numpy array): The image we are comparing to the first one

    Returns:
        numpy array: image comparison data
    """
    mse = skimage.metrics.mean_squared_error(image1, image2)
    psnr = skimage.metrics.peak_signal_noise_ratio(image1, image2)
    ssim = skimage.metrics.structural_similarity(image1, image2)
    return numpy.array([[mse, psnr, ssim]])

@_time
def normalize_to_range(array, endpoint):
    """Normalizes all of the data in a numpy array to [0, endpoint]

    Args:
        array (numpy): a numpy array
        endpoint (float): the upper range of the normalized data
    """
    array = array.astype('float32') 
    array *= (endpoint/array.max())
    return array

@_time
def breakiamge_toPatches(image, patchshape):

    patch_list = []
    img_height, img_width = numpy.shape(image)
    patch_height, patch_width = patchshape

    patch_array = image.reshape(img_height // patch_height,
                                patch_height,
                                img_width // patch_width,
                                patch_width)
    patch_array = patch_array.swapaxes(1,2)
    print(numpy.shape(patch_array))

    for i in patch_array:
        for j in i:
            temp = numpy.expand_dims(j, axis = -1)
            temp = numpy.expand_dims(temp, axis = 0)
            patch_list.append(temp)


    return patch_list



# Upscale
@_time
def predict(model, filename, downsample, scale):
    """Using the CNN to upscale the image

    Args:
        model (keras model): The loaded keras model
        filename (string): Name of the image to upscale
        downsample (bool): Determines if we are downscaling the image or not
        scale (int): The scale that we are upscaling too. Either 2 or 4
    """
    count = 0
    outPatchRows = inPatchRows - offset * 2
    outPatchCols = inPatchCols - offset * 2
    nnList = numpy.empty((0,3))
    blList = numpy.empty((0,3))
    bcList = numpy.empty((0,3))
    reconList = numpy.empty((0,3))

    xprint('Processing file "%s"...' % (filename))
    (basename, ext) = os.path.splitext(filename)

    # Open the image and covnert it to grayscale and 12-bit and save it
    gtImage = skimage.io.imread(filename)
    gtImage = skimage.img_as_uint(skimage.color.rgb2gray(skimage.color.rgba2rgb(gtImage))) & 0xFFF0
    gtImage = skimage.img_as_float(gtImage)

    # saveFileName = '%s/%s_gtImage.png' % (outputDir, basename)
    #skimage.io.imsave(saveFileName, gtImage)

    smallImage = gtImage
    downsampleIndicator = 'x'

    # Downsample image for CNN comparison if enabled
    if downsample:
        # Generate "low resolution" image 
        smallImage = skimage.transform.downscale_local_mean(gtImage, (scale, scale))
        downsampleIndicator = ''
        saveFileName = '%s/%s_%s%d_downSample.png' % (outputDir, basename, downsampleIndicator, scale)
        skimage.io.imsave(saveFileName, smallImage)
    

    # Create variabels for image reconstruction
    inRows = smallImage.shape[0] * scale
    inCols = smallImage.shape[1] * scale
    outRows = inRows - offset * 2
    outCols = inCols - offset * 2


    # -------------------Upscale using Bicubic ---------------------
    starttime_BC = time.process_time()
    bcImage = cv2.resize(smallImage, (inCols, inRows), interpolation = cv2.INTER_CUBIC)
    # bcImage = (bcImage - numpy.min(bcImage))/numpy.ptp(bcImage) # normalize to [0,1]
    saveFileName = '%s/%s_%s%d_bcImage.png' % (outputDir, basename, downsampleIndicator, scale)
    xprint("Time to upscale to Bi-Cubic = %f" % (time.process_time() - starttime_BC))
    #skimage.io.imsave(saveFileName, bcImage)
    print("Before shape")
    print(numpy.shape(bcImage))

    if downsample:
        # ------------------- Upsample using nearest neighbour interpolation ---------------------
        nnImage = cv2.resize(smallImage, (inCols, inRows), interpolation = cv2.INTER_NEAREST)
        saveFileName = '%s/%s_%s%d_nnImage.png' % (outputDir, basename, downsampleIndicator, scale)
        #skimage.io.imsave(saveFileName, nnImage)
        nnList = numpy.append(nnList, DetermineComparisons(gtImage[offset:offset+outRows, offset:offset+outCols], nnImage[offset:offset+outRows, offset:offset+outCols]), axis=0)
        # ------------------------------------------------------------------------------------

        # ------------------- Upsample using bilinear interpolation ---------------------
        saveFileName = '%s/%s_%s%d_blImage.png' % (outputDir, basename, downsampleIndicator, scale)
        blImage = cv2.resize(smallImage, (inCols, inRows), interpolation = cv2.INTER_LINEAR)
        #skimage.io.imsave(saveFileName, blImage)
        blList = numpy.append(blList, DetermineComparisons(gtImage[offset:offset+outRows, offset:offset+outCols], blImage[offset:offset+outRows, offset:offset+outCols]), axis=0)
        # ------------------------------------------------------------------------------------

        # Compare the GT with the bicubic
        bcList = numpy.append(bcList, DetermineComparisons(gtImage[offset:offset+outRows, offset:offset+outCols], bcImage[offset:offset+outRows, offset:offset+outCols]), axis=0)


    # Scan across in patches to reconstruct the full image
    r = 0
    outImage = numpy.zeros((outRows, outCols))
    CNNTime = time.process_time()

    patch_image = breakiamge_toPatches(bcImage, (128,128))

    # break image into patches and upscale then put back together
    # NOTE: this only works if the shape of the image array rows*colums is evenly divisable by 128*128
    with tf.device('/gpu:0'):
        # while r < outRows:
        #     if r + outPatchRows > outRows:
        #         r = outRows - outPatchRows
        #     c = 0
        #     while c < outCols:
        #         if c + outPatchCols > outCols:
        #             c = outCols - outPatchCols
        #         inPatch = bcImage[r:r + inPatchRows, c:c + inPatchCols]
        #         inPatch = inPatch.reshape(1, inPatchRows, inPatchCols, 1)
        result = model.predict(
            numpy.vstack(patch_image), 
            batch_size=len(patch_image)
        )
            #     temp = result[0, :, : , 0]
            #     print(numpy.shape(temp))
            #     outImage[r:r + outPatchRows, c:c + outPatchCols] = result[0, :, : , 0]
            #     c = c + outPatchCols
            # r = r + outPatchRows

    # Save the CNN image
    rows = numpy.zeros((1,1))
    cols = numpy.zeros((1,1))
    count = 0
    for i in result:
        patch = numpy.squeeze(i, axis=2) # remove the extra dimansion to make it (116,116)

        # if rows is a zeros array set it to patch and move forward in the loop
        if rows.sum() == 0:
            rows = patch
            continue
        if count < 15:
            rows = numpy.hstack((rows, patch)) # Connect all the patches on the same row together
            count += 1
        
        if count == 15:
            # if col is a zeros array copy rows to cols and create a new rows array
            if cols.sum() == 0:
                cols = numpy.copy(rows)
                rows = numpy.zeros((1,1))
            # Add the rows to the column and create a empty rows array
            else:
                cols = numpy.vstack((cols, rows))
                rows = numpy.zeros((1,1))
            count = 0

    final_image = cols
    print(numpy.shape(final_image))
    # skimage.io.imsave("TEST.png", final_image)

    # saveFileName = '%s/%s_%s%d_sisr.png' % (outputDir, basename, downsampleIndicator, scale)
    # outImage[outImage > 1] = 1
    xprint('Time to upscale using CNN = %f' % (time.process_time() - CNNTime))
    skimage.io.imsave(saveFileName, final_image)

    # Compare reconstructed image to groundtruth image
    if downsample:
        reconList = numpy.append(reconList, DetermineComparisons(gtImage[offset:offset+outRows, offset:offset+outCols], outImage), axis=0)
        xprint('     NearNeighour    Bi-Linear     Bi-Cubic  Reconstruct   Difference')
        xprint('MSE  %12.6f %12.6f %12.6f %12.6f %12.6f' % 
        (nnList[count-1,0], blList[count-1,0], bcList[count-1,0], reconList[count-1,0], reconList[count-1,0] - bcList[count-1,0]))
        xprint('PSNR %12.6f %12.6f %12.6f %12.6f %12.6f' % 
        (nnList[count-1,1], blList[count-1,1], bcList[count-1,1], reconList[count-1,1], reconList[count-1,1] - bcList[count-1,1]))
        xprint('SSIM %12.6f %12.6f %12.6f %12.6f %12.6f' % 
        (nnList[count-1,2], blList[count-1,2], bcList[count-1,2], reconList[count-1,2], reconList[count-1,2] - bcList[count-1,2]))
    



if __name__ == '__main__':

    # Log the start info
    xprint('')
    xprint(delimiterMajor)
    xprint('Project: CMPT819 SISR - Predict')
    xprint(' Author: Andrew Kostiuk')
    xprint('   Date: %s' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    xprint(delimiter)

    # load the imputs and the global variables such as model name and file name
    startTimeX = time.process_time()
    xprint(delimiter)
    modelName, filename, downsample, scale = Load_inputs()

    # Load CNN
    xprint('Using model = ' + modelName)
    model = load_model(modelName, compile=False)
    model.summary()
    xprint(delimiter)
    xprint("Time to load model and set up upscaling parameters = %f" % (time.process_time() - startTimeX))

    # Upscale the image
    predict(model, filename, downsample, scale)

    xprint('Total Time = %f' % (time.process_time() - startTimeX))
    xprint(delimiterMajor)

    log.close()



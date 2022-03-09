#************************************************************
# CMPT819 Project SISR - Predict
# Andrew Kostiuk, andrew.kostiuk@usask.ca
# December 3, 2017
# Copyright (c) 2017, Andrew Kostiuk
#------------------------------------------------------------

# All the imports we need
from pathlib import Path
import warnings
import time
import numpy
import os
import datetime

# Machine learning libraries
import keras
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# image manipulation libraries

# Original
import skimage.io
import skimage.transform
import skimage.color
import cv2

keras.backend.set_learning_phase(0)

# Allows to dynamically expand memory for Tensorflow to use
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#------------------------------------------------------------
# Settings
#------------------------------------------------------------

# Turn off warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

INPATCHROWS = 128
INPATCHCOLS = 128
MODELDIR = 'models'
OUTPUTDIR = 'upscaledImages'

N1 = 64
N2 = 32
F1 = 9
F2 = 1
F3 = 5
OFFSET = int(F1/2) + int(F3/2)

TRAININGPATCHES = 1024

DELIMITERMAJOR = '============================================================'
DELIMITER = '----------------------------------------------------------'

os.environ["CUDA_VISIBLE_DEVICES"]="0"   

local_device_protos = device_lib.list_local_devices()
([x.name for x in local_device_protos if x.device_type == 'GPU'], 
 [x.name for x in local_device_protos if x.device_type == 'CPU'])

os.environ["CUDA_VISIBLE_DEVICES"]="0"   
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
tf.config.optimizer.set_experimental_options(
    {'debug_stripper': True,
    'constant_folding': True,
    'loop_optimization': True,
    'scoped_allocator_optimization': True,
    'arithmetic_optimization': True}
)
keras.backend.set_learning_phase(0)

#---------------------TensorRT testing---------------------------
# PRECISION = "FP16"
# from helper import ModelOptimizer # using the helper from NVIDIA
# pre_model = tf.keras.models.load_model("models/model_4_t1024_e1000.h5")
# pre_model.save("test_model")
# model_dir = 'models/model_4_t1024_e1000.h5'
# opt_model = ModelOptimizer('testing_TensorRT/test_model')
# model_fp16 = opt_model.convert('testing_TensorRT/test_model'+'_FP16', precision=PRECISION)
# test_model = tf.keras.models.load_model("testing_TensorRT/test_model")
#-----------------------------------------------------------------


# Make directories if necessary
if not os.path.isdir(OUTPUTDIR):
    os.mkdir(OUTPUTDIR)

if not os.path.isdir("ExtractedFiles"):
    os.mkdir("ExtractedFiles")

# Open the loggin file
log = open('./sisrPredict.log', 'a+')
#------------------------------------------------------------


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



# Compares two images together and return MSE, PSNR, MSE
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



# Generates patches from the given image
# The pathces are in 128x128 strided by 116x116
def extract_patches(image):
    patch_size = [1,128,128,1]
    image = numpy.expand_dims(image, axis = 0)
    image = numpy.expand_dims(image, axis = -1)
    patches =  tf.image.extract_patches(image ,patch_size, [1, 116,116, 1], [1, 1, 1, 1], 'SAME')
    _, row, col, _ = patches.shape
    return row, col, tf.reshape(patches, [row*col,128,128,1])


# Upscale
def predict(model, filename, img, downsample, scale, total_image):
    """Using the CNN to upscale the image

    Args:
        model (keras model): The loaded keras model
        filename (string): Name of the image to upscale
        downsample (bool): Determines if we are downscaling the image or not
        scale (int): The scale that we are upscaling too. Either 2 or 4
    """

    xprint('')
    xprint(DELIMITERMAJOR)
    xprint('Project: SISR - Upscaler')
    xprint('   Date: %s' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    xprint('scale: %s' % (scale))
    xprint(DELIMITER)

    # Create empty arrays for image comparison
    nnList = numpy.empty((0,3))
    blList = numpy.empty((0,3))
    bcList = numpy.empty((0,3))
    reconList = numpy.empty((0,3))



    # If the image doesn't have the proper dimensions create them
    if len(img.shape) != 3:
        img = numpy.expand_dims(img, axis=-1)
        img = numpy.repeat(img, 4, axis=-1)

    # Open the image and covnert it to grayscale and 12-bit and save it
    if img.shape[-1] == 4:
        gtImage = skimage.img_as_uint(skimage.color.rgb2gray(skimage.color.rgba2rgb(img))) & 0xFFF0
    else:
         gtImage = skimage.img_as_uint(skimage.color.rgb2gray(img)) & 0xFFF0
    gtImage = skimage.img_as_float(gtImage)

    smallImage = gtImage
    downsampleIndicator = 'x'
    # Downsample image for CNN comparison if enabled
    if downsample=="True":
        # Generate "low resolution" image 
        smallImage = skimage.transform.downscale_local_mean(gtImage, (scale, scale))
        downsampleIndicator = ''
        saveFileName = '%s/%s_%s%d_downSample.png' % (OUTPUTDIR, filename, downsampleIndicator, scale)
    

    # Create variabels for image reconstruction
    inRows = smallImage.shape[0] * scale
    inCols = smallImage.shape[1] * scale
    outRows = inRows - OFFSET * 2
    outCols = inCols - OFFSET * 2
    

    # -------------------Upscale using Bicubic ---------------------
    starttime_BC = time.time() 
    bcImage = cv2.resize(smallImage, (inCols, inRows), interpolation = cv2.INTER_CUBIC)
    saveFileName = '%s/%s_%s%d_bcImage.png' % (OUTPUTDIR, filename, downsampleIndicator, scale)
    bcImageSave = bcImage*255
    cv2.imwrite(saveFileName, bcImageSave) 
    xprint("Time to upscale to Bi-Cubic = %f" % (time.time()  - starttime_BC))


    if downsample=="True":
        # ------------------- Upsample using nearest neighbour interpolation ---------------------
        nnImage = cv2.resize(smallImage, (inCols, inRows), interpolation = cv2.INTER_NEAREST)
        saveFileName = '%s/%s_%s%d_nnImage.png' % (OUTPUTDIR, filename, downsampleIndicator, scale)
        #skimage.io.imsave(saveFileName, nnImage)

        nnImageSave = nnImage*255
        cv2.imwrite(saveFileName, nnImageSave) 

        nnList = numpy.append(nnList, DetermineComparisons(gtImage[OFFSET:OFFSET+outRows, OFFSET:OFFSET+outCols], nnImage[OFFSET:OFFSET+outRows, OFFSET:OFFSET+outCols]), axis=0)
        # ------------------------------------------------------------------------------------

        # ------------------- Upsample using bilinear interpolation ---------------------
        saveFileName = '%s/%s_%s%d_blImage.png' % (OUTPUTDIR, filename, downsampleIndicator, scale)
        blImage = cv2.resize(smallImage, (inCols, inRows), interpolation = cv2.INTER_LINEAR)
        #skimage.io.imsave(saveFileName, blImage)
        blList = numpy.append(blList, DetermineComparisons(gtImage[OFFSET:OFFSET+outRows, OFFSET:OFFSET+outCols], blImage[OFFSET:OFFSET+outRows, OFFSET:OFFSET+outCols]), axis=0)
        # ------------------------------------------------------------------------------------

        blImageSave = blImage*255
        cv2.imwrite(saveFileName, blImageSave) 

        # Compare the GT with the bicubic
        bcList = numpy.append(bcList, DetermineComparisons(gtImage[OFFSET:OFFSET+outRows, OFFSET:OFFSET+outCols], bcImage[OFFSET:OFFSET+outRows, OFFSET:OFFSET+outCols]), axis=0)


    # Scan across image and break it into patches
    row_recon, col_recon, patch_test = extract_patches(bcImage)

    # break image into patches and upscale then put back together
    # NOTE: this only works if the shape of the image array rows*colums is evenly divisable by 128*128
    tim1 = time.time()
    if total_image>1:
        with tf.device('/gpu:0'):
            result = model.predict(
                # numpy.vstack(test_list), 
                patch_test,
                #verbose = 1,
                # steps = 1,
                # batch_size=len(patch_image)
            )

    else:
        with tf.device('/cpu:0'):
            result = model.predict(
                # numpy.vstack(test_list), 
                patch_test,
                #verbose = 1,
                # steps = 1,
                # batch_size=len(patch_image)
            )
    
    # Put the patches back in the proper order
    count = 0
    final_matrix = []
    inner = []
    for i in result:
        if count < col_recon:
            inner.append(numpy.squeeze(i, axis=2))
            count += 1
        if count >= col_recon:
            final_matrix.append(inner.copy())
            inner = []
            count = 0
        
    # Reconstruct the image
    final_image = numpy.array(final_matrix)
    print(final_image.shape)
    numrows, numcols, height, width = numpy.shape(final_matrix)
    final_image = final_image.reshape(numrows, numcols, height, width).swapaxes(1, 2).reshape(height*numrows, width*numcols, 1)
    print(final_image.shape)

    # Remove the black border
    finalRowStart = (final_image.shape[0]-outRows)//2
    finalColStart = (final_image.shape[1]-outCols)//2
    finalRowEnd = int(final_image.shape[0] - finalRowStart)
    finalColEnd = int(final_image.shape[1] - finalColStart)
    final_image = final_image[finalRowStart:finalRowEnd,finalColStart:finalColEnd]

    filename = Path(filename).stem
    saveFileName = '%s/%s_%s%d_sisr.png' % (OUTPUTDIR, filename, downsampleIndicator, scale)
    final_image[final_image > 1] = 1
    final_image[final_image < 0] = 0

    final_image = final_image.astype("float64") # We need this for image comparison 
    xprint('Time to upscale using CNN = %f' % (time.time()-tim1))

    #  Save the final image

    final_imageSave = final_image*255
    cv2.imwrite(saveFileName, final_imageSave) 

    # Compare reconstructed image to groundtruth image
    if downsample=="True":
        reconList = numpy.append(reconList, DetermineComparisons(gtImage[OFFSET:OFFSET+outRows, OFFSET:OFFSET+outCols], final_image.reshape(final_image.shape[0],final_image.shape[1])), axis=0)
        xprint('     NearNeighour    Bi-Linear     Bi-Cubic  Reconstruct   Difference')
        xprint('MSE  %12.6f %12.6f %12.6f %12.6f %12.6f' % 
        (nnList[count-1,0], blList[count-1,0], bcList[count-1,0], reconList[count-1,0], reconList[count-1,0] - bcList[count-1,0]))
        xprint('PSNR %12.6f %12.6f %12.6f %12.6f %12.6f' % 
        (nnList[count-1,1], blList[count-1,1], bcList[count-1,1], reconList[count-1,1], reconList[count-1,1] - bcList[count-1,1]))
        xprint('SSIM %12.6f %12.6f %12.6f %12.6f %12.6f' % 
        (nnList[count-1,2], blList[count-1,2], bcList[count-1,2], reconList[count-1,2], reconList[count-1,2] - bcList[count-1,2]))
        
        return nnList[count-1,0], blList[count-1,0], bcList[count-1,0], reconList[count-1,0], reconList[count-1,0] - bcList[count-1,0], nnList[count-1,1], blList[count-1,1], bcList[count-1,1], reconList[count-1,1], reconList[count-1,1] - bcList[count-1,1], nnList[count-1,2], blList[count-1,2], bcList[count-1,2], reconList[count-1,2], reconList[count-1,2] - bcList[count-1,2]

        # comparison.write('%12.6f,%12.6f,%12.6f,%12.6f,%12.6f\n' % 
        # (nnList[count-1,0], blList[count-1,0], bcList[count-1,0], reconList[count-1,0], reconList[count-1,0] - bcList[count-1,0]))
        # comparison.write('%12.6f,%12.6f,%12.6f,%12.6f,%12.6f\n' % 
        # (nnList[count-1,1], blList[count-1,1], bcList[count-1,1], reconList[count-1,1], reconList[count-1,1] - bcList[count-1,1]))
        # comparison.write('%12.6f,%12.6f,%12.6f,%12.6f,%12.6f\n' % 
        # (nnList[count-1,2], blList[count-1,2], bcList[count-1,2], reconList[count-1,2], reconList[count-1,2] - bcList[count-1,2]))
        # comparison.close()




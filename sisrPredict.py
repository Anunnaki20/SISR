#************************************************************
# CMPT819 Project SISR - Predict
# Andrew Kostiuk, andrew.kostiuk@usask.ca
# December 3, 2017
# Copyright (c) 2017, Andrew Kostiuk
#------------------------------------------------------------

#************************************************************
# Turn off warnings
#------------------------------------------------------------
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#************************************************************
# Settings
#------------------------------------------------------------
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

scale = 0
trainingPatches = 1024

#************************************************************
# Make directories if necessary
#------------------------------------------------------------
import os
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)
    
#************************************************************
# Logging functions
#------------------------------------------------------------
f = open(outputDir + '/sisrPredict.log', 'a+')
def xprint(string):
    string = (datetime.datetime.now().strftime("%H:%M:%S")) + ' ' + string
    print(string)
    f.write(string + '\n')
    f.flush()
    return

import datetime
    
delimiterMajor = '============================================================'
delimiter = '------------------------------------------------------------'
xprint(delimiterMajor)
xprint('Project: CMPT819 SISR - Predict')
xprint(' Author: Andrew Kostiuk')
xprint('   Date: %s' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
xprint(delimiter)

#************************************************************
# Get input file
#------------------------------------------------------------
import sys
if len(sys.argv) < 4:
    xprint('Usage: python sisrPredict.py <input_image> -s <2|4> [-d]')
    xprint(delimiterMajor)
    quit()
    
filename = sys.argv[1]
if not os.path.isfile(filename):
    xprint('Input file "%s" not found.' % (filename))
    xprint(delimiterMajor)
    quit()

if sys.argv[2] != '-s':
    xprint('Second parameter "%s" not scale.' % (sys.argv[2]))
    xprint(delimiterMajor)
    quit()

scale = int(sys.argv[3])
if scale != 2 and scale != 4:
    xprint('Invalid scale = %d.' % (scale))
    xprint(delimiterMajor)
    quit()

downsample = False
if len(sys.argv) > 4:
    if sys.argv[4] != '-d':
        xprint('Invalid parameter "s".' % (sys.argv[4]))
        xprint(delimiterMajor)
        quit()
    else:
        downsample = True
        
xprint('Parameters:')
xprint('    scale = %d' % (scale))
xprint('    trainingPatches = %d' % (trainingPatches))
if downsample:
    downsampleString = 'True'
else:
    downsampleString = 'False'
xprint('    downsample = %s' % (downsampleString))
xprint(delimiter)

modelName = '%s/model_%d_t%03d_e1000.h5' % (modelDir, scale, trainingPatches)
if not os.path.isfile(modelName):
    xprint('Model "%s" not found.' % (modelName))
    xprint(delimiterMajor)
    quit()
    
#************************************************************
# Import libraries
#------------------------------------------------------------
import time
startTimeX = time.process_time()
xprint('Importing libraries:')

import skimage.io
import skimage.transform
import skimage.color
import numpy
from keras.models import load_model

xprint('Time = %f' % (time.process_time() - startTimeX))
xprint(delimiter)

#************************************************************
# Load CNN
#------------------------------------------------------------
xprint('Using model = ' + modelName)
model = load_model(modelName)
model.summary()
xprint(delimiter)
    
#************************************************************
# Helper functions
#------------------------------------------------------------
def DetermineComparisons(image1, image2):
    mse = skimage.measure.compare_mse(image1, image2)
    psnr = skimage.measure.compare_psnr(image1, image2)
    ssim = skimage.measure.compare_ssim(image1, image2)
    return numpy.array([[mse, psnr, ssim]])
    
#************************************************************
# Predict
#------------------------------------------------------------
xprint('Predicting...')
startTime = time.process_time()

count = 0
outPatchRows = inPatchRows - offset * 2
outPatchCols = inPatchCols - offset * 2
nnList = numpy.empty((0,3))
blList = numpy.empty((0,3))
bcList = numpy.empty((0,3))
reconList = numpy.empty((0,3))

xprint('Processing file "%s"...' % (filename))
(basename, ext) = os.path.splitext(filename)

# Get ground truth image and convert to 'effectively' 12-bit grayscale
gtImage = skimage.img_as_float(skimage.io.imread(filename))
gtImage = skimage.img_as_uint(skimage.color.rgb2gray(gtImage)) & 0xFFF0
gtImage = skimage.img_as_float(gtImage)
saveFileName = '%s/%s_gtImage.png' % (outputDir, basename)
skimage.io.imsave(saveFileName, gtImage)
    
if downsample:
    # Generate "low resolution" image 
    smallImage = skimage.transform.downscale_local_mean(gtImage, (scale, scale))
    downsampleIndicator = ''
    saveFileName = '%s/%s_%s%d_downSample.png' % (outputDir, basename, downsampleIndicator, scale)
    skimage.io.imsave(saveFileName, smallImage)
else:
    smallImage = gtImage
    downsampleIndicator = 'x'
    
# Upsample using nearest neighbour interpolation
inRows = smallImage.shape[0] * scale
inCols = smallImage.shape[1] * scale
outRows = inRows - offset * 2
outCols = inCols - offset * 2
nnImage = skimage.transform.resize(smallImage, (inRows, inCols), order=0)
saveFileName = '%s/%s_%s%d_nnImage.png' % (outputDir, basename, downsampleIndicator, scale)
skimage.io.imsave(saveFileName, nnImage)
if downsample:
    nnList = numpy.append(nnList, DetermineComparisons(gtImage[offset:offset+outRows, offset:offset+outCols], nnImage[offset:offset+outRows, offset:offset+outCols]), axis=0)
    
# Upsample using bilinear interpolation
blImage = skimage.transform.resize(smallImage, (inRows, inCols), order=1)
saveFileName = '%s/%s_%s%d_blImage.png' % (outputDir, basename, downsampleIndicator, scale)
skimage.io.imsave(saveFileName, blImage)
if downsample:
    blList = numpy.append(blList, DetermineComparisons(gtImage[offset:offset+outRows, offset:offset+outCols], blImage[offset:offset+outRows, offset:offset+outCols]), axis=0)

# Upsample using bicubic interpolation
bcImage = skimage.transform.resize(smallImage, (inRows, inCols), order=3)
saveFileName = '%s/%s_%s%d_bcImage.png' % (outputDir, basename, downsampleIndicator, scale)
skimage.io.imsave(saveFileName, bcImage)
if downsample:
    bcList = numpy.append(bcList, DetermineComparisons(gtImage[offset:offset+outRows, offset:offset+outCols], bcImage[offset:offset+outRows, offset:offset+outCols]), axis=0)
   
# Scan across in patches to reconstruct the full image
outImage = numpy.zeros((outRows, outCols))
r = 0
while r < outRows:
    if r + outPatchRows > outRows:
        r = outRows - outPatchRows
    c = 0
    while c < outCols:
        if c + outPatchCols > outCols:
            c = outCols - outPatchCols
        inPatch = bcImage[r:r + inPatchRows, c:c + inPatchCols]
        inPatch = inPatch.reshape(1, inPatchRows, inPatchCols, 1)
        result = model.predict(
            inPatch, 
            batch_size=32,
            verbose=0
        )
        outImage[r:r + outPatchRows, c:c + outPatchCols] = result[0, :, : , 0]
        c = c + outPatchCols
    r = r + outPatchRows
            
saveFileName = '%s/%s_%s%d_sisr.png' % (outputDir, basename, downsampleIndicator, scale)
outImage[outImage > 1] = 1
skimage.io.imsave(saveFileName, outImage)

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
        
xprint('Time = %f' % (time.process_time() - startTime))
xprint(delimiter)

#************************************************************
# Program end
#------------------------------------------------------------
xprint('Total Time = %f' % (time.process_time() - startTimeX))
xprint(delimiterMajor)
f.close()

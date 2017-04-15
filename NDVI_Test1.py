from __future__ import print_function

import numpy as np #arrays and math
import cv2 #opencv library
import sys

#-------------------------------------------
#----------------NDVI Function--------------
#-------------------------------------------

#NDVI Calculation
#Input: an RGB image frame from infrablue source (blue is blue, red is pretty much infrared)
#Output: an RGB frame with equivalent NDVI of the input frame
def NDVICalc(original):
    "This function performs the NDVI calculation and returns an RGB frame)"
    lowerLimit = 5 #this is to avoid divide by zero and other weird stuff when color is near black

    #First, make containers
    oldHeight,oldWidth = original[:,:,0].shape;
    ndviImage = np.zeros((oldHeight,oldWidth,3),np.uint8) #make a blank RGB image
    ndvi      = np.zeros((oldHeight,oldWidth  ),np.int  ) #make a blank b/w image for storing NDVI value
    red       = np.zeros((oldHeight,oldWidth  ),np.int  ) #make a blank array for red
    blue      = np.zeros((oldHeight,oldWidth  ),np.int  ) #make a blank array for blue

    #Now get the specific channels. Remember: (B , G , R)
    red  = (original[:,:,2]).astype('float')
    blue = (original[:,:,0]).astype('float')

    #Perform NDVI calculation
    summ                  = red+blue
    summ[summ<lowerLimit] = lowerLimit #do some saturation to prevent low intensity noise

    ndvi = (((red-blue)/(summ)+1)*127).astype('uint8')  #the index

    redSat            = (ndvi-128)*2        # red channel
    bluSat            = ((255-ndvi)-128)*2  # blue channel
    redSat[ndvi<128]  = 0;                  # if the NDVI is negative, no red info
    bluSat[ndvi>=128] = 0;                  # if the NDVI is positive, no blue info


    #And finally output the image. Remember: (B , G , R)
    ndviImage[:,:,2] = redSat                # Red Channel
    ndviImage[:,:,0] = bluSat                # Blue Channel
    ndviImage[:,:,1] = 255-(bluSat+redSat)   # Green Channel

    return ndviImage;


#-------------------------------------------
#----------------DVI Function---------------
#-------------------------------------------

#DVI Calculation
#Input: an RGB image frame from infrablue source (blue is blue, red is pretty much infrared)
#Output: an RGB frame with equivalent DVI of the input frame
def DVICalc(original):
    "This function performs the DVI calculation and returns an RGB frame)"

    #First, make containers
    oldHeight,oldWidth = original[:,:,0].shape;
    dviImage = np.zeros((oldHeight,oldWidth,3),np.uint8) #make a blank RGB image
    dvi      = np.zeros((oldHeight,oldWidth  ),np.int  ) #make a blank b/w image for storing DVI value
    red      = np.zeros((oldHeight,oldWidth  ),np.int  ) #make a blank array for red
    blue     = np.zeros((oldHeight,oldWidth  ),np.int  ) #make a blank array for blue

    #Now get the specific channels. Remember: (B , G , R)
    red  = (original[:,:,2]).astype('float')
    blue = (original[:,:,0]).astype('float')

    #Perform DVI calculation
    dvi = (((red-blue)+255)/2).astype('uint8')  #the index

    redSat           = (dvi-128)*2       #red channel
    bluSat           = ((255-dvi)-128)*2 #blue channel
    redSat[dvi<128]  = 0;                #if the NDVI is negative, no red info
    bluSat[dvi>=128] = 0;                #if the NDVI is positive, no blue info


    #And finally output the image. Remember: (B , G , R)
    #Red Channel
    dviImage[:,:,2] = redSat

    #Blue Channel
    dviImage[:,:,0] = bluSat

    #Green Channel
    dviImage[:,:,1] = 255-(bluSat+redSat)

    return dviImage;

# https://github.com/robintw/RPiNDVI/blob/master/ndvi.py

def disp_multiple(im1=None, im2=None, im3=None, im4=None):
    """
    Combines four images for display.
    """
    height, width = im1.shape
    combined      = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)

    combined[0:height, 0:width, :] = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
    combined[ height:, 0:width, :] = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    combined[0:height,  width:, :] = cv2.cvtColor(im3, cv2.COLOR_GRAY2RGB)
    combined[ height:,  width:, :] = cv2.cvtColor(im4, cv2.COLOR_GRAY2RGB)

    return combined


def label(image, text):
    """
    Labels the given image with the given text
    """
    return cv2.putText(image, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)


def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-95%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out  = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out



#-------------------------------------------
#----------------Main Function--------------
#-------------------------------------------


vFile = '/home/ubuntu/xchange/Plant1.mov'
#vFile = '/home/ubuntu/xchange/Garden2.avi'
#vFile = '/home/ubuntu/xchange/GreenWhite.mov'

fourcc = cv2.cv.CV_FOURCC(*'XVID')
vc     = cv2.VideoCapture(vFile)
n      = 0
flag   = True

if vc.isOpened(): # try to get the first frame
    print('Video File:' + vFile + ' Oppened')
    rval, frame = vc.read()

    if flag:
        flag        = False
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        print('Running on OpenCV Version:{0}.{1}.{2}'.format(major_ver, minor_ver, subminor_ver))

        # With webcam get(CV_CAP_PROP_FPS) does not work.
        if int(major_ver) < 3: fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
        else:                  fps = vc.get(cv2.CAP_PROP_FPS)

        height, width = frame.shape[:2]
        fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
        print('Video Size:' + str(width)+ 'x' + str(height)+ ' @ ' + str(fps) + ' FPS')
        x           = int(   width/2)       #Text Related
        y           = int(2*height/3)
        text_color  = (255,255,255)         #color as (B,G,R)
        font        = cv2.FONT_HERSHEY_PLAIN
        thickness   = 2
        font_size   = 2.0
        vn          = cv2.VideoWriter('outputPlant1a.avi', fourcc, fps, (int(width*2), int(height*2)))  # FPS, Res: 640x426, 720x483, 1280x720, 1920x1080
        vd          = cv2.VideoWriter('outputPlant1b.avi', fourcc, fps, (int(width), int(height)))  # FPS, Res: 720x483, 1280x720, 1920x1080
else:
    rval        = False
    print("NO DATA")

#if rval:
while rval:
    # Get the individual colour components of the image
    b, g, r = cv2.split(frame)

    # Calculate the NDVI

    # Bottom of fraction
    bottom              = (r.astype(float) + b.astype(float))
    bottom[bottom == 0] = 0.01  # Make sure we don't divide by zero!

    ndvi = (r.astype(float) - b) / bottom
    ndvi = contrast_stretch(ndvi)
    ndvi = ndvi.astype(np.uint8)

    # Do the labelling
    label(   b,  'Blue')
    label(   g, 'Green')
    label(   r,   'NIR')
    label(ndvi,  'NDVI')

    # Combine ready for display
    combined = disp_multiple(b, g, r, ndvi)
    vn.write(combined)
    #vd.write(frame)
    #cv2.imwrite("reultn.jpg", combined)

    ndviImage   = NDVICalc(frame)
    ##dviImage    = DVICalc( frame)

    #cv2.putText(frame,     "Raw Image" , (x,y), font, font_size, text_color, thickness, lineType=cv2.CV_AA)
    cv2.putText(ndviImage, "NDVI Image", (x,y), font, font_size, text_color, thickness, lineType=cv2.CV_AA)
    ##cv2.putText(dviImage,  "DVI Image" , (x,y), font, font_size, text_color, thickness, lineType=cv2.CV_AA)
    ##vn.write(ndviImage)
    vd.write( ndviImage)
    #newFrame = np.concatenate((ndviImage,dviImage,frame),axis=1)

    rval, frame = vc.read()
    n +=1
    #print(' '+str(n)+' ', end="\r", flush=True)
    print('## Processing Frame:'+str(n)+' ', end="\r")
    sys.stdout.flush()

# When everything done, release the capture
print("End of file {0} frames processed".format(n))
vc.release()
vn.release()
vd.release()


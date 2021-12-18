# Import packages
import cv2
import numpy as np
import pytesseract
import pyttsx3

def ocr(img):
    """
    -takes an image
    -processes the image
    -performs OCR on the processed image
    -returns mask and text
    """
    
    # BEGIN TESSERACT
    # Up-sample
    img = cv2.resize(img, (0, 0), fx=2, fy=2)

    # Convert to greyscale
    # greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to HSV color-space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
 
    # Get the binary mask
    # mask = cv2.inRange(greyscale, 0, 62)
    mask = cv2.inRange(hsv, np.array([0, 0, 90]), np.array([179, 255, 255]))

    #invert the binary mask
    mask = cv2.bitwise_not(mask)

#     kernel = np.ones((2,2), np.uint8)
#     mask = cv2.erode(img, kernel, iterations = 4)

    # OCR
    txt = pytesseract.image_to_string(mask, lang='eng')
    txt = txt.replace('\n', ' ')
    txt = txt.replace('', '') # removing a weird non printable character

    print(txt)
    
    return mask, txt

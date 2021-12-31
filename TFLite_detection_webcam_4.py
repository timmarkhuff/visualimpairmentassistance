######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from datetime import datetime

# import more packages
import pytesseract
import ocr
import pyttsx3
import pyttsx3_functions
import gpiozero

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30): # (self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
        
# Define and parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.7)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1920x1080') # original '1280x720'
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--showvideo', help='Display the video on screen. Disable this for better performance',
                    default=1)

parser.add_argument('--testmode', help='Calibrated for test signs. Disable for real Safeway signs.',
                    default=1)

args = parser.parse_args()

# GLOBAL VARIABLES
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
SHOW_VIDEO = int(args.showvideo)
TEST_MODE = int(args.testmode)
RUN = True
SHORT_PRESS = False
RETRY_ATTEMPTS = 0

min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# GENERAL FUNCTIONS
def write_to_text(text):
    
    text = str(text)
    with open('screenshots/log.txt', 'a') as file:
        file.writelines(text)
        file.writelines('\n')
        
def add_column_headers():
    """
    check if log.txt already has column headers
    if not, add them                
    """     
        
    column_headers = "DateTime,DetectedObjects,ObjectWidth,ObjectHeight,ObjectArea,DewarpTime,OCRTime,DetectedText"
    with open('screenshots/log.txt', 'r') as original: data = original.read()
    if len(data) > 0:       
        if data[0] != "D":
            with open('screenshots/log.txt', 'w') as modified: modified.write(column_headers + data)
    elif len(data) == 0:
        with open('screenshots/log.txt', 'w') as modified: modified.write(column_headers + data)

def handle_button(): 
    def thread():
        global RUN, SHORT_PRESS, SHOW_VIDEO, TEST_MODE
        sampling_rate = .1 # in seconds
        short_press_duration = (.1, 1) # a range of seconds
        long_hold_duration = 6 # seconds
        array_length = int(np.round(long_hold_duration / sampling_rate))
        array = np.array(np.zeros(array_length, dtype=int))
        array_one_quarter = int(np.round(array_length * .25))
        array_one_half = int(np.round(array_length * .5))
        array_three_quarters = int(np.round(array_length * .75))
    
        # define the physical button for the device
        button = gpiozero.Button(17)
        while RUN:
            if button.is_pressed:
                array = np.concatenate((np.ones(1, dtype=int), array[0:-1]), axis=0)
            else:
                array = np.concatenate((np.zeros(1, dtype=int), array[0:-1]), axis=0)
            # print(array)
            
            # check if the button has been released
            if array[0] == 0 and array[1] == 1: 
                press_samples = 1
                for index, i in enumerate(array[:-2]):
                    if i == 1 and array[index + 1] == 1:
                        press_samples += 1
                    elif i == 1 and array[index + 1] == 0:
                      break
                    
                upper_cycles = short_press_duration[1] / sampling_rate
                lower_cycles = short_press_duration[0] / sampling_rate
                
                # check the duration for which the button was held
                # perform certain actions accordingly
                if lower_cycles < press_samples < upper_cycles:
                    SHORT_PRESS = True
                    # speak to the user when the image processing begins
                    pyttsx3_functions.text_to_speech("Scanning...")
                if array_one_quarter <= press_samples < array_one_half:
                    if SHOW_VIDEO:
                        SHOW_VIDEO = False
                        pyttsx3_functions.text_to_speech("Video disabled.")
                        cv2.destroyAllWindows()
                    else:
                        SHOW_VIDEO = True
                        pyttsx3_functions.text_to_speech("Video enabled.")
                if array_one_half <= press_samples:
                    if TEST_MODE:
                        TEST_MODE = False
                        pyttsx3_functions.text_to_speech("Test mode disabled.")
                    else:
                        TEST_MODE = True
                        pyttsx3_functions.text_to_speech("Test mode enabled.")
                        
            # perform certain actions if the button is held for a specified duration 
            if all(array[:array_one_quarter]) and array[array_one_quarter] == 0:
                pyttsx3_functions.text_to_speech("Release button to toggle video mode.")
            elif all(array[:array_one_half]) and array[array_one_half] == 0:
                pyttsx3_functions.text_to_speech("Release button to toggle test mode.")
            elif all(array[:array_three_quarters]) and array[array_three_quarters] == 0:
                pyttsx3_functions.text_to_speech("Continue holding button to shut down.")
            elif all(array):
                pyttsx3_functions.text_to_speech("Shutting down.")
                time.sleep(1)
                os.system("poweroff")             
                
            time.sleep(sampling_rate)
                
    Thread(target=thread).start()



# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# the txt that is returned by the OCR function
txt = ""

# add column headers to log.txt
add_column_headers()

# welcome message
pyttsx3_functions.text_to_speech("Welcome to VIA. Press the button on your device to scan for signs.")

# initiate handle button function
handle_button()

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while RUN:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    unmarked_frame = frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    
    detected_object_list = []
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)): 

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            objwidth = xmax - xmin
            objheight = ymax - ymin
            objarea = objwidth * objheight
            
                   
            # draw rectangle
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            detected_object = [object_name, ymin, xmin, ymax, xmax, objwidth, objheight, objarea]
            detected_object_list.append(detected_object)
            
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    frame_to_save = frame.copy()
    
    # draw detected text
    if len(txt) < 3:
        txt = "(No text detected. Press the button to scan for signs.)"
    cv2.putText(frame, f'{txt}',(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    if SHOW_VIDEO:
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('VIA', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # check if key is pressed
    pressed_key = cv2.waitKey(1)
    
    # DETECT LARGEST OBJECT
    if pressed_key == ord('d') or pressed_key == ord('D') or SHORT_PRESS:
        
        # capture a new time and date stamp when button is pressed
        # skip this part if we are re-attempting. This will give
        # us an accurate time score
        if RETRY_ATTEMPTS == 0:
            dateTimeObjStart = datetime.utcnow()
            timestampStr = dateTimeObjStart.strftime("%Y.%m.%d.%H:%M.%S.%f")
        else:
            pass
                                 
        areas = []
        
        # make a list of the areas of all detected objects
        if detected_object_list != []:
            # speak to the user when the image processing begins
            # pyttsx3_functions.text_to_speech("Reading sign...") # this seems to be slowing the process down
            for object in detected_object_list:
                areas.append(object[-1])
                
            # calculate the maximum area of all detected objects
            max_area = max(areas)
                        
            # label each object as largest(1) or not largest(0)
            for i in range(len(detected_object_list)):
                if areas[i] == max_area:
                    #save the coordinates of the largest object
                    largest_ymin = detected_object_list[i][1] 
                    largest_xmin = detected_object_list[i][2]  
                    largest_ymax = detected_object_list[i][3] 
                    largest_xmax = detected_object_list[i][4]
                                        
                    # get cropped image
                    cropped_image = frame1[largest_ymin:largest_ymax, largest_xmin:largest_xmax]
                    
                    # dewarp
                    dewarped, dewarp_process = ocr.process_and_unwarp(cropped_image, test_mode=TEST_MODE)
                    
                    # calculate time elapsed from button press to dewarp completion
                    dateTimeObjEndDewarp = datetime.utcnow()
                    time_elapsed_dewarp = (dateTimeObjEndDewarp - dateTimeObjStart).total_seconds()
                    
                    # OCR 
                    mask, ocr_dict = ocr.darius_ocr_v2(dewarped)
                    
                    # parse OCR dictionary, remove low confidence words and add
                    # semicolons to end of lines
                    txt = ocr.parse_ocr_dict(ocr_dict)
                    
                    # calculate time elapsed from button press to OCR completion
                    dateTimeObjEndOCR = datetime.utcnow()
                    time_elapsed_ocr = (dateTimeObjEndOCR - dateTimeObjStart).total_seconds()
                    
                    # Text to speech with PYTTSX3
                    pyttsx3_functions.text_to_speech(txt)
                    
                    # reset these variabls
                    SHORT_PRESS = False
                    RETRY_ATTEMPTS = 0
                                     
        else:
            if RETRY_ATTEMPTS <= 3:
                print("No signs detected. Reattempting...")
                RETRY_ATTEMPTS += 1
            else:
                pyttsx3_functions.text_to_speech("Sorry, I don't see any signs. Please scan again.")
                # set this flag back to False so that the button can be pressed again.
                SHORT_PRESS = False
                RETRY_ATTEMPTS = 0
        
        
        
        # draw detected text on the screen
        if len(txt) < 3:
            txt = "(No text detected. Press 'd' to detect text.)"
        cv2.putText(frame_to_save, f'{txt}',(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # save whole image
        image = frame_to_save
        file_path = f'screenshots/{timestampStr}_whole.png'
        ocr.save_image(image, file_path)
        
        if detected_object_list != []:
                        
            # save the dewarp process image
            image = dewarp_process
            file_path = f'screenshots/{timestampStr}_dewarp_process.png'
            ocr.save_image(image, file_path)

            # save mask
            image = mask
            file_path = f'screenshots/{timestampStr}_mask.png'
            ocr.save_image(image, file_path)

            
        if RETRY_ATTEMPTS == 0:
            # write the results to log.txt
            if len(detected_object_list) == 0:
                time_elapsed_dewarp = 0
                time_elapsed_ocr = 0
                txt = ""
                obj_width = 0
                obj_height = 0
                obj_area = 0
            else:
                obj_width = cropped_image.shape[0]
                obj_height = cropped_image.shape[1]
                obj_area = obj_width * obj_height
            
            write_to_text(f'{timestampStr},{len(detected_object_list)},'\
                          f'{obj_width},{obj_height},{obj_area},'\
                          f'{time_elapsed_dewarp},{time_elapsed_ocr},"{txt}"')
    
            
    if pressed_key == ord('q') or pressed_key == ord('Q'):
        pyttsx3_functions.text_to_speech("Goodbye. Thank you for using Via!")
      
            
        # Clean up
        RUN = False
        cv2.destroyAllWindows()
        videostream.stop()
        break
                


    

            
        

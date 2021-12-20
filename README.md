## Installing Visual Impairment Assistance on a Raspberry Pi

### Getting Started
* install Raspberry Pi OS on your SD drive. Follow the instructions here: https://www.raspberrypi.com/documentation/computers/getting-started.html
* Follow the tutorial from Edje Electronics to get a simple computer vision model running on your Raspberry Pi. Video: https://www.youtube.com/watch?v=aimSGOAUI8Y&t. Written guide: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md
* On your Raspberry Pi, download the files from this Github repository (visualimpairmentassistance).
* From the files you just downloaded, move the following files/folders in the tflite1 folder on your Raspberry Pi. 
  * screenshots (the program will save screenshots and mp3s here. There is also log.txt which records diagnostic information about each time the button is pressed)
  * ocr.py (this contains the code for Optical Character Recognition)
  * pyttsx3_functions.py (this is for Text to Speech)
  * TFLite_detection_webcam_4.py (this is the main script)
  * models (this contains several different Object Recognition models that you can choose from)
  * launch_commands (these are the commands you will run in the terminal to launch the program)

### Install Packages
All of these packages must be installed in the virtual environment, tflite1-env. To activate the virtual environment, run these lines in the terminal:
* cd tflite1
* source tflite1-env/bin/activate

Run the following command lines in the terminal of the Raspberry Pi to install the necessary packages. Some of these might need to be adapted for Linux. If you get any errors, try googling the proper way to install the package on Linux.
* pip install kornia
* sudo apt install tesseract-ocr -y
* sudo apt install libtesseract-dev -y
* pip install pytesseract

### Deploying the model
* Go to the models folder and copy the model you wish to use. We will probably start with actual_safeway_signs_v2_tradeoff.tflite. We can experiment with the others if we find this one doesn't work well.
* Go to the Sample_TF-Lite_model folder, delete the file entitled detect.tflite, paste the model that you copied on the previous step, and rename it detect.tflite.
* Open the lablelmap.txt file. Delete everything and simply write the word 'sign'. Sign is the only class that we need to detect. Save and close the file.

### Starting the Application
* Open the Raspberry Pi terminal and run all of the command lines found in the launch_commands file. You can copy and paste them and run them all at once. You have to run the program in this way, because it needs to run in the virtual environment. The python script can't be executed directly. 
* Pressing d on the keyboard will cause the program to detect signs (we will need to replace this with the physical button later)
* Presing q causes the program to quit. You can also just close the terminal window.

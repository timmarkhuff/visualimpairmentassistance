## Installing Visual Impairment Assistance on a Raspberry Pi

### Getting Started
* install Raspberry Pi OS on your SD drive. Follow the instructions here: https://www.raspberrypi.com/documentation/computers/getting-started.html
* Follow the tutorial from Edje Electronics to get a simple computer vision model running on your Raspberry Pi. Video: https://www.youtube.com/watch?v=aimSGOAUI8Y&t. Written guide: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md
* On your Raspberry Pi, download the files from this Github repository (visualimpairmentassistance).
* Move all of the files and folders that you just downloaded from Github to the tflite1 folder.

### Install Packages
All of these packages must be installed in the virtual environment, tflite1-env. To activate the virtual environment, run these lines in the terminal:
* cd tflite1
* source tflite1-env/bin/activate

Run the following command lines in the terminal of the Raspberry Pi to install the necessary packages. Some of these might need to be adapted for Linux. If you get any errors, try googling the proper way to install the package on Linux.
* sudo apt update
* pip install pytesseract
* sudo apt install tesseract-ocr -y
* sudo apt install libtesseract-dev -y
* pip install pyttsx3
* sudo apt-get update && sudo apt-get install espeak
* pip install gpiozero
* sudo pip3 install pyrebase

### Loading the OCR Model
We have trained a custom OCR model to recognize the Helvetica Bold text of Safeway signs. This file needs to be moved into a particular folder in order to work.
* in the command prompt, issue sudo mv (file path of the Helvetica Bold OCR model on your Raspberry Pi) /usr/share/tesseract-ocr/4.00/tessdata

### Starting the Application
* Open the Raspberry Pi terminal and run all of the command lines found in the launch_commands file. You can copy and paste them and run them all at once. You have to run the program in this way, because it needs to run in the virtual environment. The python script can't be executed directly. 
* Pressing d on the keyboard will cause the program to detect signs (we will need to replace this with the physical button later)
* Presing q causes the program to quit. You can also just close the terminal window.

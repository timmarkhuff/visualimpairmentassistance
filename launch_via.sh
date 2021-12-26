#!/bin/bash
cd tflite1
. tflite1-env/bin/activate
python3 TFLite_detection_webcam_4.py --modeldir=test_model_v2 --testmode=1 --showvideo=1

# INSTALL THESE PACKAGES
# pip install wheel
# pip install pyttsx3
# sudo apt-get update && sudo apt-get install espeak

import pyttsx3

engine = pyttsx3.init()

def text_to_speech(timestamp, txt):
    if len(str(txt)) < 5:
        txt = "No text detected."
    else:
        pass
        
    engine.say(txt)
    engine.save_to_file(txt, f"screenshots/{timestamp}.mp3")
    engine.runAndWait()

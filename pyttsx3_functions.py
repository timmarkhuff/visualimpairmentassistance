# INSTALL THESE PACKAGES
# pip install wheel
# pip install pyttsx3
# sudo apt-get update && sudo apt-get install espeak

import pyttsx3

engine = pyttsx3.init()

def text_to_speech(txt):
    if len(str(txt)) < 3:
        txt = "No text detected."
    else:
        pass
    
    print(f"Speaking: {txt}")
    engine.say(txt)
    
    # # saves an mp3 to the Raspberry Pi's memory
    # timestamp = timestampStr = dateTimeObjStart.strftime("%Y.%m.%d.%H:%M.%S.%f")
    # engine.save_to_file(txt, f"screenshots/{timestamp}.mp3")
    
    engine.runAndWait()
    

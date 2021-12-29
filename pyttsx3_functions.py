# INSTALL THESE PACKAGES
# pip install wheel
# pip install pyttsx3
# sudo apt-get update && sudo apt-get install espeak

import pyttsx3
import threading

engine = pyttsx3.init()
engine.setProperty('voice', "english_rp")

def text_to_speech(txt):
    if len(str(txt)) < 3:
        txt = "I'm sorry. I can't read that text. Please try again."
    else:
        pass
    
    try:
        print(f"Speaking: {txt}")
        engine.say(txt)
        engine.runAndWait()
    except:
        print(f"Speaking (suppressed): {txt}")
    
    # # saves an mp3 to the Raspberry Pi's memory
    # timestamp = timestampStr = dateTimeObjStart.strftime("%Y.%m.%d.%H:%M.%S.%f")
    # engine.save_to_file(txt, f"screenshots/{timestamp}.mp3")


if __name__ == "__main__":
    text_to_speech("Chips; Candy; Nuts; Snacks; Protein Bars; Granola;")
    run_and_wait()




    
    

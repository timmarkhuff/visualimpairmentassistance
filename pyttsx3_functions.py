# INSTALL THESE PACKAGES
# pip install wheel
# pip install pyttsx3
# sudo apt-get update && sudo apt-get install espeak

import pyttsx3

engine = pyttsx3.init()

def text_to_speech(txt):
    if len(str(txt)) < 3:
        txt = "I'm sorry. I can't read that text. Please try again."
    else:
        pass
    
    print(f"Speaking: {txt}")
    engine.setProperty('voice', "english_rp")
    engine.say(txt)
    
    # # saves an mp3 to the Raspberry Pi's memory
    # timestamp = timestampStr = dateTimeObjStart.strftime("%Y.%m.%d.%H:%M.%S.%f")
    # engine.save_to_file(txt, f"screenshots/{timestamp}.mp3")
    
    engine.runAndWait()
    
if __name__ == "__main__":
    text_to_speech("Chips; Candy; Nuts; Snacks; Protein Bars; Granola;")
    
    

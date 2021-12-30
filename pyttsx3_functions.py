# INSTALL THESE PACKAGES
# pip install wheel
# pip install pyttsx3
# sudo apt-get update && sudo apt-get install espeak

import pyttsx3
from threading import Thread

engine = pyttsx3.init()
engine.setProperty('voice', "english_rp")

def text_to_speech(txt):
    def thread(txt):
        # count the number of valid characters in the string
        characters = 0
        for i in txt:
            if i.isalnum():
                characters += 1
                
        if characters < 4:
            txt = "I'm sorry. I can't read that text. Please try again."
        else:
            pass
        
        try:
            print(f"Speaking: {txt}")
            engine.say(txt)
            engine.runAndWait()
        except:
            print(f"Speaking (suppressed): {txt}")
        
    Thread(target=thread(txt)).start()
    
if __name__ == "__main__":
    text_to_speech("Chips; Candy; Nuts; Snacks; Protein Bars; Granola;")




    
    

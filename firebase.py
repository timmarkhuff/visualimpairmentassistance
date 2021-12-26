# https://www.youtube.com/watch?v=cAU5qMCw9Bo
# https://www.youtube.com/watch?v=I1eskLk0exg
import pyrebase
import os

config = {
  "apiKey": "AIzaSyA3l-OyTRWphW2L-h3FoByqHEdu3nbxml8",
  "authDomain": "visual-impairment-assistance.firebaseapp.com",
  "databaseURL": "https://visual-impairment-assistance-default-rtdb.firebaseio.com",
  "projectId": "visual-impairment-assistance",
  "storageBucket": "visual-impairment-assistance.appspot.com",
  "messagingSenderId": "201527576702",
  "appId": "1:201527576702:web:41760af74d684fefa8b02f",
  "measurementId": "G-84Y9W41H22"
};

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
database = firebase.database()

# send images
print("Sending images...")
arr = os.listdir("screenshots")
for i in arr:
    
    
    path_on_cloud = f"screenshots/{i}"
    path_local = f"screenshots/{i}"
    storage.child(path_on_cloud).put(path_local)
print("Images sent!")

# send data
print("Sending data...")

dict = {}
with open("screenshots/log.txt", "r") as a_file:
    for line in a_file:
        stripped_line = line.strip() # strips the end-line break from each line

        list = stripped_line.split(",")
        key = list[0]
          
        key = key.replace(".", "")
        key = key.replace(":", "")
        key = key.replace(" ", "")

        dict[key] = list
 
database.child("Visual Impairment Assistance")
data = dict
database.update(data)
    
print("Data sent!")
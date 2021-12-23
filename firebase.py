import pyrebase

config = {
  apiKey: "AIzaSyA3l-OyTRWphW2L-h3FoByqHEdu3nbxml8",
  authDomain: "visual-impairment-assistance.firebaseapp.com",
  databaseURL: "https://visual-impairment-assistance-default-rtdb.firebaseio.com",
  projectId: "visual-impairment-assistance",
  storageBucket: "visual-impairment-assistance.appspot.com",
  messagingSenderId: "201527576702",
  appId: "1:201527576702:web:41760af74d684fefa8b02f",
  measurementId: "G-84Y9W41H22"
};

firebase = pyrebase.initialize_app(config)

storage = firebase.storage()
database = firebase.database()
a = "Hello World"
print (a)
database.child("Visual Impairment Assistance")
data = {"key1": a}
database.set(data)

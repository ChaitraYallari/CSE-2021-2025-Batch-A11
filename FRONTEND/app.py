from flask import Flask, url_for, redirect, render_template, request, session
import mysql.connector, os
from flask import *
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16, InceptionV3, VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
import cv2
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import os
import base64
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import joblib


UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='bc'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return render_template('home.html')
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

feature_extractor = load_model('feature_extractor_model.h5')
ensemble_model = joblib.load('ensemble_model.joblib')
class_names = ['adenocarcinoma', 'benign', 'squamous_carcinoma']

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img_resized = cv2.resize(img, target_size) 
    img_resized = np.expand_dims(img_resized, axis=0)  
    img_resized = img_resized / 255.0  
    return img_resized

def extract_features_from_image(img_path):
    img_preprocessed = preprocess_image(img_path)
    features = feature_extractor.predict(img_preprocessed)
    return features


def predict_image(img_path):
    features = extract_features_from_image(img_path)
    features = features.reshape(1, -1)
    prediction = ensemble_model.predict(features)

    return prediction


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', prediction_result=None)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', prediction_result=None)
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            predicted_label = prediction[0]
            predicted_class_name = class_names[predicted_label]
            with open(file_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            prediction_result = {'prediction': predicted_class_name, 'image': image_data}
            
            return render_template('upload.html', prediction_result=prediction_result)

    return render_template('upload.html', prediction_result=None)

    
if __name__ == "__main__":
    app.run(debug=True)







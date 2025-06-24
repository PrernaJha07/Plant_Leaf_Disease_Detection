from flask import Flask, render_template, request, redirect, url_for, session
import os, json
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil

app = Flask(__name__)
app.secret_key = 'plant123'
UPLOAD_FOLDER = 'ml/uploads'
MODEL_PATH = 'ml/model.h5'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(MODEL_PATH)
classes = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'PlantVillage',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',
    'no_plant'
]

cures = {
    "Pepper__bell___Bacterial_spot": "Use copper-based bactericides. Remove infected leaves and improve air circulation.",
    "Pepper__bell___healthy": "No disease detected. Maintain proper watering and sunlight.",
    "PlantVillage": "This seems like a general folder name, please upload a specific leaf image.",
    "Potato___Early_blight": "Apply fungicides such as chlorothalonil and practice crop rotation.",
    "Potato___Late_blight": "Use resistant varieties and fungicides like mancozeb or metalaxyl.",
    "Potato___healthy": "No disease detected. Keep good farming practices.",
    "Tomato_Bacterial_spot": "Apply bactericides, remove infected foliage, and use disease-free seeds.",
    "Tomato_Early_blight": "Use crop rotation and fungicides such as chlorothalonil.",
    "Tomato_Late_blight": "Use resistant varieties and fungicides like mancozeb or metalaxyl.",
    "Tomato_Leaf_Mold": "Improve ventilation and apply fungicides such as copper-based sprays.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves and apply fungicides. Practice crop rotation.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray insecticidal soap or neem oil and maintain humidity.",
    "Tomato__Target_Spot": "Remove infected leaves and apply appropriate fungicides.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Remove infected plants and control whitefly vectors.",
    "Tomato__Tomato_mosaic_virus": "Use resistant varieties and disinfect tools regularly.",
    "Tomato_healthy": "No disease detected. Maintain good growing conditions.",
    "no_plant": "The uploaded image does not appear to be a plant leaf. Please upload a valid plant image."
}


@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        with open('users.json', 'r') as f:
            users = json.load(f)
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users:
            return "User already exists"
        users[uname] = pwd
        with open('users.json', 'w') as f:
            json.dump(users, f)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        with open('users.json', 'r') as f:
            users = json.load(f)
        uname = request.form['username']
        pwd = request.form['password']
        if users.get(uname) == pwd:
            session['user'] = uname
            return redirect(url_for('index'))
        else:
            return "Invalid credentials"
    return render_template('login.html')

@app.route('/forgot', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        uname = request.form['username']
        with open('users.json', 'r') as f:
            users = json.load(f)
        if uname in users:
            return f"Your password is: {users[uname]}"
        else:
            return "Username not found"
    return render_template('forgot.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None
    cure = None
    filename = None

    if request.method == 'POST':
        file = request.files.get('leaf')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            static_path = os.path.join('static/uploads', filename)
            os.makedirs('static/uploads', exist_ok=True)
            shutil.copy(path, static_path)

            # Load and preprocess the image
            img = image.load_img(path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict with model
            result = model.predict(img_array)
            predicted_index = np.argmax(result)

            # Safety check for predicted index in classes
            if predicted_index < len(classes):
                predicted_class = classes[predicted_index]
                prediction = predicted_class
                cure = cures.get(predicted_class, "No cure information available.")
            else:
                prediction = "Unknown"
                cure = "Prediction out of range, please try another image."

    return render_template('index.html', prediction=prediction, cure=cure, username=session['user'], filename=filename)

if __name__ == '__main__':
    app.run(debug=True)

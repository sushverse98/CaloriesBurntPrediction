import os
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify

# Tell Flask where the templates (HTML) are located
template_dir = os.path.abspath('../frontend')  # Assuming you're running app from backend folder

app = Flask(__name__, template_folder=template_dir)

# Load the trained model
model = None
try:
    with open('calories_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Ensure 'calories_model.pkl' is in backend folder.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch form data
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        duration = int(request.form['duration'])
        heartrate = int(request.form['heartrate'])
        body_temp = float(request.form['body_temp'])

        # Prepare data for prediction
        input_data = np.array([[gender, age, height, weight, duration, heartrate, body_temp]])

        # Make prediction
        prediction = model.predict(input_data)

        return jsonify({'prediction': f"{prediction[0]:.2f} kcal"})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)

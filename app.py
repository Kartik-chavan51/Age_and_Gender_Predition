from flask import Flask, render_template, request, url_for
import numpy as np
import os
from werkzeug.utils import secure_filename
from model import image_pre, predict  # Ensure this file exists

app = Flask(__name__)

# Define static folder explicitly
BASE_DIR = r'D:\DS projects\Ageand Gender prediciton\age+gender+prediction+Project+Code\Project Code\app'
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static')
ALLOWED_EXTENSIONS = {'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure static folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None  # Initialize result for GET requests
    
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'No file uploaded. Please select a file.'
        
        file1 = request.files['file1']
        
        if file1.filename == '':
            return 'No file selected. Please upload a PNG file.'
        
        if file1 and allowed_file(file1.filename):
            filename = secure_filename('input.png')  # Standard name for uploaded file
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file1.save(path)
            
            # Process image and make prediction
            data = image_pre(path)
            
            if data is None:
                return "Error: Image processing failed. Please check the input."
            
            try:
                age, gen = predict(data)
                gender = "Male" if gen == 1 else "Female"
                result = f'Predicted age is {age} years and the person is {gender}'
            except Exception as e:
                result = f"Prediction error: {str(e)}"
        else:
            result = 'Invalid file type. Only PNG files are allowed.'
    
    # Pass logo path explicitly for debugging (optional)
    logo_path = url_for('static', filename='logo1.jpg')
    print(f"Logo path: {logo_path}")  # Check this in terminal
    
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
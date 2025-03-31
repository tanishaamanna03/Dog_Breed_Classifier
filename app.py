from flask import Flask, render_template, request, jsonify
import os
from model import DogBreedClassifier
from werkzeug.utils import secure_filename
import torch
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("Initializing model...")
classifier = DogBreedClassifier()

def get_latest_model():
    model_dir = 'models'
    if not os.path.exists(model_dir):
        return None
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return None
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return os.path.join(model_dir, latest_model)

latest_model = get_latest_model()
if latest_model:
    print(f"Loading model from: {latest_model}")
    classifier.load_model(latest_model)
else:
    print("No trained model found. Please train the model first.")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            start_time = time.time()
            prediction = classifier.predict(filepath)
            prediction_time = time.time() - start_time

            prediction['processing_time'] = f"{prediction_time:.2f}s"
            
            os.remove(filepath)
            return jsonify(prediction)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 
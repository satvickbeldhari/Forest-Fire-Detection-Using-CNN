from flask import Flask, request, render_template
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('fire_detection_model.h5')

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to model's expected input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return render_template('result.html', prediction="Error: No image uploaded", image_path=None)
        
        file = request.files['image']
        img = Image.open(file.stream)
        
        # Save the image temporarily
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(image_path)
        
        # Preprocess and predict
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        label = "No Fire Detected" if prediction[0][0] > 0.5 else "Fire Detected"
        
        return render_template('result.html', prediction=label, image_path=f"uploads/{filename}")
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", image_path=None)

if __name__ == '__main__':
    app.run(debug=True)


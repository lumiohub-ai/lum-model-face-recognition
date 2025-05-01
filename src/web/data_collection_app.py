from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('yolov8m-face.pt')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure data/images directory exists
if not os.path.exists('data/images'):
    os.makedirs('data/images')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})

    file = request.files['file']
    name = request.form.get('name', '')

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})

    if not name:
        return jsonify({'success': False, 'message': 'Please enter a name'})

    if file and allowed_file(file.filename):
        try:
            # Read the image file
            file_bytes = np.fromstring(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Detect and crop face
            yolo_results = model(image, max_det=1)

            x1, y1, x2, y2, _, _ = yolo_results[0].boxes.data[0].cpu().numpy()

            face_image = image[int(y1):int(y2), int(x1):int(x2)]

            # save image to the folder
            cv2.imwrite(f'data/images/{name}.jpg', face_image)

            if face_image is None:
                return jsonify({'success': False, 'message': 'No face detected in the image'})

            return jsonify({
                'success': True,
                'message': 'Face detected and saved successfully',
                'filename': f'{name}.jpg'
            })

        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error processing image: {str(e)}'
            })

    return jsonify({'success': False, 'message': 'Invalid file type'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)
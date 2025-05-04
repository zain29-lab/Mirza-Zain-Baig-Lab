from flask import Flask, render_template, request, jsonify
import os
from ultralytics import YOLO
import cv2
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO("yolov8n.pt")  


DEFAULT_LOCATION = {'lat': 34.0522, 'lon': -118.2437}  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    filename = str(uuid.uuid4()) + '.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    results = model(filepath)[0]

    
    animal_ids = [16, 17, 18, 20, 21]  
    count = sum(1 for r in results.boxes.cls if int(r.item()) in animal_ids)

    if count >= 3:
        alert = {
            "lat": DEFAULT_LOCATION['lat'],
            "lon": DEFAULT_LOCATION['lon'],
            "count": count,
            "message": f"Animal herd detected! ({count} animals)"
        }
        return jsonify(alert)
    else:
        return jsonify({"message": "No herd detected."})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    app.run(debug=True)

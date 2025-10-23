from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import mediapipe as mp  # Removed to avoid DLL issues
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Initialize MediaPipe (disabled to avoid DLL issues)
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

class ElephantIdentifier:
    def __init__(self):
        self.model = None
        self.elephant_data = {}
        self.feature_extractor = FeatureExtractor()
        
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def extract_features_from_image(self, image_path):
        """Extract features from image using multiple methods"""
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        features = {}
        
        # 1. Edge Detection Features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features['edge_orientation'] = self._calculate_edge_orientation(edges)
        
        # 2. Color Features (grayscale statistics)
        features['mean_gray'] = np.mean(gray)
        features['std_gray'] = np.std(gray)
        features['skewness'] = self._calculate_skewness(gray)
        
        # 3. Texture Features
        features['texture_energy'] = self._calculate_texture_energy(gray)
        features['texture_contrast'] = self._calculate_texture_contrast(gray)
        
        # 4. Shape Features
        features['shape_complexity'] = self._calculate_shape_complexity(edges)
        features['aspect_ratio'] = self._calculate_aspect_ratio(edges)
        
        # 5. Pose Estimation Features (if applicable)
        pose_features = self._extract_pose_features(image)
        features.update(pose_features)
        
        return features
    
    def _calculate_edge_orientation(self, edges):
        """Calculate dominant edge orientation"""
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(sobely, sobelx)
        return np.mean(orientation)
    
    def _calculate_skewness(self, image):
        """Calculate skewness of pixel intensities"""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        return np.mean(((image - mean) / std) ** 3)
    
    def _calculate_texture_energy(self, image):
        """Calculate texture energy using local binary patterns"""
        # Simplified texture energy calculation
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)
        return np.mean(filtered ** 2)
    
    def _calculate_texture_contrast(self, image):
        """Calculate texture contrast"""
        return np.std(image)
    
    def _calculate_shape_complexity(self, edges):
        """Calculate shape complexity based on edge density"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        if area == 0:
            return 0
        return (perimeter ** 2) / (4 * np.pi * area)
    
    def _calculate_aspect_ratio(self, edges):
        """Calculate aspect ratio of the main object"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 1
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return w / h if h > 0 else 1
    
    def _extract_pose_features(self, image):
        """Extract pose estimation features (simplified version without MediaPipe)"""
        features = {}
        # Simplified pose features without MediaPipe
        try:
            # Use basic image analysis for pose-like features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Calculate center of mass as a proxy for pose
            moments = cv2.moments(gray)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                features['pose_confidence'] = 1.0
                features['head_position'] = cy / height  # normalized y position
                features['body_orientation'] = abs(cx - width/2) / (width/2)  # horizontal offset
            else:
                features['pose_confidence'] = 0.0
                features['head_position'] = 0.5
                features['body_orientation'] = 0.0
        except:
            features['pose_confidence'] = 0.0
            features['head_position'] = 0.5
            features['body_orientation'] = 0.0
        
        return features
    
    def _calculate_body_orientation(self, landmarks):
        """Calculate body orientation based on key landmarks (simplified)"""
        # Simplified body orientation calculation without MediaPipe landmarks
        return 0.0  # Placeholder value
    
    def train_model(self, training_data):
        """Train the elephant identification model"""
        if not training_data:
            return False, "No training data provided"
        
        X = []
        y = []
        
        for elephant_name, features_list in training_data.items():
            for features in features_list:
                if features:
                    feature_vector = list(features.values())
                    X.append(feature_vector)
                    y.append(elephant_name)
        
        if len(X) == 0:
            return False, "No valid features extracted"
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Save model
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'elephant_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        return True, f"Model trained successfully with {len(X)} samples"
    
    def predict(self, image_path):
        """Predict elephant identity from image"""
        if self.model is None:
            return None, "Model not trained yet"
        
        features = self.extract_features_from_image(image_path)
        if features is None:
            return None, "Could not extract features from image"
        
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        prediction = self.model.predict(feature_vector)[0]
        confidence = np.max(self.model.predict_proba(feature_vector))
        
        return prediction, confidence

class FeatureExtractor:
    def __init__(self):
        pass

# Initialize the elephant identifier
elephant_identifier = ElephantIdentifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    elephant_name = request.form.get('elephant_name', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not elephant_identifier.allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    if not elephant_name:
        return jsonify({'error': 'Elephant name is required'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Extract features
    features = elephant_identifier.extract_features_from_image(filepath)
    
    if features is None:
        return jsonify({'error': 'Could not process image'}), 400
    
    # Store features
    if elephant_name not in elephant_identifier.elephant_data:
        elephant_identifier.elephant_data[elephant_name] = []
    
    elephant_identifier.elephant_data[elephant_name].append(features)
    
    # Convert image to base64 for thumbnail
    image_base64 = None
    try:
        with open(filepath, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    except:
        pass
    
    return jsonify({
        'success': True,
        'message': f'File uploaded and features extracted for {elephant_name}',
        'filename': filename,
        'features_count': len(elephant_identifier.elephant_data[elephant_name]),
        'thumbnail': image_base64
    })

@app.route('/upload_multiple', methods=['POST'])
def upload_multiple_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    elephant_name = request.form.get('elephant_name', '')
    
    if not elephant_name:
        return jsonify({'error': 'Elephant name is required'}), 400
    
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected'}), 400
    
    uploaded_files = []
    failed_files = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if not elephant_identifier.allowed_file(file.filename):
            failed_files.append(f"{file.filename} - Invalid file type")
            continue
        
        try:
            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract features
            features = elephant_identifier.extract_features_from_image(filepath)
            
            if features is None:
                failed_files.append(f"{file.filename} - Could not process image")
                try:
                    os.remove(filepath)
                except:
                    pass
                continue
            
            # Store features
            if elephant_name not in elephant_identifier.elephant_data:
                elephant_identifier.elephant_data[elephant_name] = []
            
            elephant_identifier.elephant_data[elephant_name].append(features)
            
            # Convert image to base64 for thumbnail
            image_base64 = None
            try:
                with open(filepath, 'rb') as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            except:
                pass
            
            uploaded_files.append({
                'filename': filename,
                'original_name': file.filename,
                'thumbnail': image_base64
            })
            
        except Exception as e:
            failed_files.append(f"{file.filename} - {str(e)}")
    
    return jsonify({
        'success': True,
        'message': f'Uploaded {len(uploaded_files)} files for {elephant_name}',
        'uploaded_files': uploaded_files,
        'failed_files': failed_files,
        'features_count': len(elephant_identifier.elephant_data[elephant_name])
    })

@app.route('/train', methods=['POST'])
def train_model():
    if not elephant_identifier.elephant_data:
        return jsonify({'error': 'No training data available'}), 400
    
    success, message = elephant_identifier.train_model(elephant_identifier.elephant_data)
    
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'error': message}), 400

@app.route('/predict', methods=['POST'])
def predict_elephant():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not elephant_identifier.allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save temporary file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"temp_{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Predict
    prediction, confidence = elephant_identifier.predict(filepath)
    
    # Convert image to base64 for display
    image_base64 = None
    try:
        with open(filepath, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    except:
        pass
    
    # Clean up temporary file
    try:
        os.remove(filepath)
    except:
        pass
    
    if prediction is None:
        return jsonify({'error': confidence}), 400
    
    return jsonify({
        'success': True,
        'prediction': prediction,
        'confidence': float(confidence),
        'image': image_base64
    })

@app.route('/data', methods=['GET'])
def get_training_data():
    return jsonify({
        'elephants': list(elephant_identifier.elephant_data.keys()),
        'total_samples': sum(len(samples) for samples in elephant_identifier.elephant_data.values()),
        'samples_per_elephant': {name: len(samples) for name, samples in elephant_identifier.elephant_data.items()}
    })

@app.route('/clear', methods=['POST'])
def clear_data():
    elephant_identifier.elephant_data = {}
    return jsonify({'success': True, 'message': 'All training data cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

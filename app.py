"""
AudioXplore Flask Backend
Advanced AI-powered audio analysis API

This backend is configured for deployment on Vercel.
It downloads models from an external source on cold start.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import librosa
import tensorflow as tf
import joblib
from werkzeug.utils import secure_filename
import warnings
import requests  # --- VERSEL MODIFICATION ---: Added for downloading models

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- VERSEL MODIFICATION START ---
# Vercel's serverless environment is ephemeral. We can only write to the /tmp directory.
# Models will be downloaded here on the first run (cold start).
MODEL_FOLDER = '/tmp/models'
UPLOAD_FOLDER = '/tmp/uploads'

# Create directories if they don't exist
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# --- VERSEL MODIFICATION END ---

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'aac', 'm4a'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global variables for models (will be loaded by the load_models function)
models = {}
scalers = {}
label_encoders = {}


# --- VERSEL MODIFICATION START ---
# This dictionary holds the direct download links for all your model files from GitHub LFS.
# IMPORTANT: You MUST replace the placeholder URLs with your actual links.
MODEL_URLS = {
    "speaker_recognition_model.h5": "https://github.com/madmax-codeman/AudioXplore-Backend-Deploy/raw/refs/heads/main/models/speaker_recognition_model.h5?download=",
    "fake_audio_detection_model.h5": "https://github.com/madmax-codeman/AudioXplore-Backend-Deploy/raw/refs/heads/main/models/fake_audio_detection_model.h5?download=",
    "emotion_detection_model.h5": "https://github.com/madmax-codeman/AudioXplore-Backend-Deploy/raw/refs/heads/main/models/emotion_detection_model.h5?download=",
    "gender_classification_model.h5": "https://github.com/madmax-codeman/AudioXplore-Backend-Deploy/raw/refs/heads/main/models/gender_classification_model.h5",
    "male_age_model.h5": "https://github.com/madmax-codeman/AudioXplore-Backend-Deploy/raw/refs/heads/main/models/male_age_model.h5?download=",
    "female_age_model.h5": "https://github.com/madmax-codeman/AudioXplore-Backend-Deploy/raw/refs/heads/main/models/female_age_model.h5?download=",
    "label_encoder.pkl": "https://github.com/madmax-codeman/AudioXplore-Backend-Deploy/raw/refs/heads/main/models/label_encoder.pkl",
    "scaler.pkl": "https://github.com/madmax-codeman/AudioXplore-Backend-Deploy/raw/refs/heads/main/models/scaler.pkl"
}

def download_file(url, destination):
    """Downloads a file from a URL to a destination path."""
    print(f"Downloading from {url} to {destination}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ Successfully downloaded {os.path.basename(destination)}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {os.path.basename(destination)}. Error: {e}")
        return False

def load_models():
    """
    Downloads models from their URLs if they don't exist in the /tmp/models directory,
    then loads them into memory.
    """
    all_files_present = True
    for filename, url in MODEL_URLS.items():
        destination_path = os.path.join(MODEL_FOLDER, filename)
        if not os.path.exists(destination_path):
            if not download_file(url, destination_path):
                all_files_present = False
    
    if not all_files_present:
        print("❌ Could not download all required model files. Aborting model loading.")
        return False

    try:
        print("Loading models from /tmp/models directory...")
        # Speaker Recognition
        models['speaker'] = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'speaker_recognition_model.h5'))
        label_encoders['speaker'] = joblib.load(os.path.join(MODEL_FOLDER, 'label_encoder.pkl'))
        
        # Fake Audio Detection
        models['fake'] = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'fake_audio_detection_model.h5'))
        scalers['fake'] = joblib.load(os.path.join(MODEL_FOLDER, 'scaler.pkl'))
        
        # Emotion Detection
        models['emotion'] = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'emotion_detection_model.h5'))
        
        # Gender Classification
        models['gender'] = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'gender_classification_model.h5'))
        models['male_age'] = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'male_age_model.h5'))
        models['female_age'] = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'female_age_model.h5'))
        
        print("✅ All models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models from files: {e}")
        return False
# --- VERSEL MODIFICATION END ---


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========== Feature Extraction Functions (No changes needed here) ==========

def extract_mfcc_speaker(audio_path):
    """Extract MFCC features for speaker identification"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfcc.shape[1] == 0:
            return None
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error extracting MFCC for speaker: {e}")
        return None

def extract_mfcc_fake(audio_path, n_mfcc=40, max_pad_len=174):
    """Extract MFCC features for fake audio detection"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc.flatten()
    except Exception as e:
        print(f"Error extracting MFCC for fake detection: {e}")
        return None

def extract_mfcc_emotion(audio_path, n_mfcc=40):
    """Extract MFCC features for emotion detection"""
    try:
        audio, sr = librosa.load(audio_path, sr=None, res_type="kaiser_fast")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting MFCC for emotion: {e}")
        return None

def extract_features_gender_age(audio_path, target_length=130):
    """Extract features for gender and age analysis"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        if log_spectrogram.shape[1] < target_length:
            pad_width = target_length - log_spectrogram.shape[1]
            log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        elif log_spectrogram.shape[1] > target_length:
            log_spectrogram = log_spectrogram[:, :target_length]
        
        return log_spectrogram
    except Exception as e:
        print(f"Error extracting features for gender/age: {e}")
        return None

# ========== API Endpoints (No changes needed here) ==========

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0,
        'available_endpoints': [
            '/analyze/speaker',
            '/analyze/emotion', 
            '/analyze/fake',
            '/analyze/gender-age',
            '/analyze/all'
        ]
    })

@app.route('/analyze/speaker', methods=['POST'])
def analyze_speaker():
    """Speaker identification endpoint"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        mfcc_features = extract_mfcc_speaker(filepath)
        if mfcc_features is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to extract features'}), 500
        
        mfcc_features = mfcc_features[np.newaxis, ..., np.newaxis]
        prediction = models['speaker'].predict(mfcc_features)
        speaker = label_encoders['speaker'].inverse_transform([np.argmax(prediction)])[0]
        confidence = float(np.max(prediction) * 100)
        
        os.remove(filepath)
        
        return jsonify({
            'speaker': speaker,
            'confidence': confidence,
            'status': 'success'
        })
        
    except Exception as e:
        # Clean up file in case of error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze/emotion', methods=['POST'])
def analyze_emotion():
    """Emotion detection endpoint"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        emotion_dict = {
            0: 'Angry', 1: 'Calm', 2: 'Happy', 3: 'Sad',
            4: 'Neutral', 5: 'Fearful', 6: 'Disgust',
            7: 'Surprised', 8: 'Pleasant Surprise'
        }
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        mfcc_features = extract_mfcc_emotion(filepath)
        if mfcc_features is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to extract features'}), 500
        
        mfcc_features = np.reshape(mfcc_features, (1, mfcc_features.shape[0], 1))
        prediction = models['emotion'].predict(mfcc_features)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_dict[emotion_idx]
        confidence = float(prediction[0][emotion_idx] * 100)
        
        os.remove(filepath)
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'status': 'success'
        })
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze/fake', methods=['POST'])
def analyze_fake():
    """Fake audio detection endpoint"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        mfcc_features = extract_mfcc_fake(filepath)
        if mfcc_features is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to extract features'}), 500
        
        scaled_features = scalers['fake'].transform([mfcc_features])
        prediction = models['fake'].predict(scaled_features).flatten()
        is_fake = prediction[0] >= 0.5
        label = "Fake" if is_fake else "Real"
        confidence = float((prediction[0] if is_fake else 1 - prediction[0]) * 100)
        
        os.remove(filepath)
        
        return jsonify({
            'classification': label,
            'confidence': confidence,
            'probability': float(prediction[0]),
            'status': 'success'
        })
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze/gender-age', methods=['POST'])
def analyze_gender_age():
    """Gender and age analysis endpoint"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        age_groups = ["child", "teen", "twenties", "thirties", "fourties", 
                     "fifties", "sixties", "seventies", "eighties"]
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        features = extract_features_gender_age(filepath)
        if features is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to extract features'}), 500
        
        features_input = features[np.newaxis, ..., np.newaxis]
        gender_pred = models['gender'].predict(features_input)
        gender = 'male' if np.argmax(gender_pred) == 0 else 'female'
        gender_confidence = float(gender_pred[0][np.argmax(gender_pred)] * 100)
        
        age_model = models['male_age'] if gender == 'male' else models['female_age']
        age_pred = age_model.predict(features_input)
        age_idx = np.argmax(age_pred)
        age_group = age_groups[age_idx]
        age_confidence = float(age_pred[0][age_idx] * 100)
        
        os.remove(filepath)
        
        return jsonify({
            'gender': gender,
            'gender_confidence': gender_confidence,
            'age_group': age_group,
            'age_confidence': age_confidence,
            'status': 'success'
        })
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze/all', methods=['POST'])
def analyze_all():
    """Complete audio analysis endpoint (all models)"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = {}
        
        # Speaker Identification
        try:
            mfcc_speaker = extract_mfcc_speaker(filepath)
            if mfcc_speaker is not None:
                mfcc_speaker = mfcc_speaker[np.newaxis, ..., np.newaxis]
                pred_speaker = models['speaker'].predict(mfcc_speaker)
                speaker = label_encoders['speaker'].inverse_transform([np.argmax(pred_speaker)])[0]
                results['speaker'] = {
                    'name': speaker,
                    'confidence': float(np.max(pred_speaker) * 100)
                }
        except Exception as e:
            results['speaker'] = {'error': str(e)}
        
        # Emotion Detection
        try:
            emotion_dict = {
                0: 'Angry', 1: 'Calm', 2: 'Happy', 3: 'Sad',
                4: 'Neutral', 5: 'Fearful', 6: 'Disgust',
                7: 'Surprised', 8: 'Pleasant Surprise'
            }
            mfcc_emotion = extract_mfcc_emotion(filepath)
            if mfcc_emotion is not None:
                mfcc_emotion = np.reshape(mfcc_emotion, (1, mfcc_emotion.shape[0], 1))
                pred_emotion = models['emotion'].predict(mfcc_emotion)
                emotion_idx = np.argmax(pred_emotion)
                results['emotion'] = {
                    'emotion': emotion_dict[emotion_idx],
                    'confidence': float(pred_emotion[0][emotion_idx] * 100)
                }
        except Exception as e:
            results['emotion'] = {'error': str(e)}
        
        # Fake Audio Detection
        try:
            mfcc_fake = extract_mfcc_fake(filepath)
            if mfcc_fake is not None:
                scaled_features = scalers['fake'].transform([mfcc_fake])
                pred_fake = models['fake'].predict(scaled_features).flatten()
                is_fake = pred_fake[0] >= 0.5
                results['fake_detection'] = {
                    'classification': "Fake" if is_fake else "Real",
                    'confidence': float((pred_fake[0] if is_fake else 1 - pred_fake[0]) * 100),
                    'probability': float(pred_fake[0])
                }
        except Exception as e:
            results['fake_detection'] = {'error': str(e)}
        
        # Gender and Age Analysis
        try:
            age_groups = ["child", "teen", "twenties", "thirties", "fourties", 
                         "fifties", "sixties", "seventies", "eighties"]
            features = extract_features_gender_age(filepath)
            if features is not None:
                features_input = features[np.newaxis, ..., np.newaxis]
                gender_pred = models['gender'].predict(features_input)
                gender = 'male' if np.argmax(gender_pred) == 0 else 'female'
                
                age_model = models['male_age'] if gender == 'male' else models['female_age']
                age_pred = age_model.predict(features_input)
                age_idx = np.argmax(age_pred)
                
                results['demographics'] = {
                    'gender': gender,
                    'gender_confidence': float(gender_pred[0][np.argmax(gender_pred)] * 100),
                    'age_group': age_groups[age_idx],
                    'age_confidence': float(age_pred[0][age_idx] * 100)
                }
        except Exception as e:
            results['demographics'] = {'error': str(e)}
        
        os.remove(filepath)
        
        return jsonify({
            'results': results,
            'status': 'success'
        })
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

# --- VERSEL MODIFICATION ---
# This part is not needed for Vercel as it handles the server.
# However, it's useful to keep for local testing.
# Vercel will ignore this when deploying.
if __name__ == '__main__':
    print("🎵 AudioXplore Backend Starting for LOCAL testing...")
    load_models()
    # For local development, you might want to use a different port than Vercel's default
    app.run(debug=True, host='0.0.0.0', port=5001)

# --- VERSEL MODIFICATION ---
# Load models on application startup for the serverless environment
load_models()

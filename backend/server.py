from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import speech_recognition as sr
import tempfile
from gtts import gTTS
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import io
import logging
import platform
import shutil
import traceback
import time
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Check for ffmpeg installation
if shutil.which("ffmpeg") is None:
    logger.warning("ffmpeg command not found in PATH. Please install ffmpeg: sudo apt-get install ffmpeg")

ACCENT_CODES = {
    "neutral": "en",
    "american": "en-us",
    "british": "en-gb",
    "australian": "en-au",
    "canadian": "en-ca"
}

def convert_to_wav(input_path):
    """Convert audio to WAV format compatible with speech_recognition"""
    try:
        temp_path = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}.wav")
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_path, format="wav")
        logger.info(f"Converted audio to WAV at {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        raise

def save_uploaded_file(file):
    """Save uploaded file with security checks"""
    try:
        if not allowed_file(file.filename):
            raise ValueError("Invalid file type")
        orig_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        orig_filepath = os.path.join(UPLOAD_FOLDER, orig_filename)
        file.save(orig_filepath)
        return convert_to_wav(orig_filepath)
    except Exception as e:
        logger.error(f"File save error: {str(e)}")
        raise

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

def transcribe_audio(audio_path):
    """Transcribe audio using multiple recognition services"""
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            
        services = [
            ("Google", recognizer.recognize_google),
            ("Sphinx", recognizer.recognize_sphinx)
        ]
        
        for service_name, recognizer_fn in services:
            try:
                text = recognizer_fn(audio_data)
                if text.strip():
                    logger.info(f"{service_name} transcription success")
                    print(text)
                    return text
            except sr.UnknownValueError:
                logger.warning(f"{service_name} couldn't understand audio")
            except sr.RequestError as e:
                logger.error(f"{service_name} error: {str(e)}")
                
        logger.error("All transcription services failed")
        return None
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return None

def blend_audio(original_path, neutralized_path, strength, output_path):
    """Blend audio signals with improved length handling"""
    try:
        y_orig, sr_orig = librosa.load(original_path, sr=None)
        y_neut, sr_neut = librosa.load(neutralized_path, sr=sr_orig)
        
        # Resample if needed
        if sr_neut != sr_orig:
            y_neut = librosa.resample(y_neut, orig_sr=sr_neut, target_sr=sr_orig)
            
        # Match lengths
        min_length = min(len(y_orig), len(y_neut))
        y_orig = y_orig[:min_length]
        y_neut = y_neut[:min_length]
        
        # Normalize and blend
        y_orig = librosa.util.normalize(y_orig)
        y_neut = librosa.util.normalize(y_neut)
        
        blended = (1 - (strength/100)) * y_orig + (strength/100) * y_neut
        
        sf.write(output_path, blended, sr_orig)
        return output_path
    except Exception as e:
        logger.error(f"Audio blending error: {str(e)}")
        raise

def scheduled_cleanup():
    """Periodic cleanup of temporary files"""
    while True:
        try:
            cleanup_old_files()
            logger.info("Scheduled cleanup completed")
            time.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

def cleanup_old_files():
    """Remove files older than 1 hour"""
    try:
        cutoff = time.time() - 3600
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, TEMP_FOLDER]:
            for filename in os.listdir(folder):
                path = os.path.join(folder, filename)
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    logger.info(f"Removed {path}")
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

@app.route('/process', methods=['POST'])
def process_audio():
    original_path = None
    neutralized_path = None
    final_path = None
    
    try:
        # Validate input
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio provided'}), 400
            
        file = request.files['audio']
        if not file or file.filename == '':
            return jsonify({'error': 'Empty file'}), 400
            
        # Process strength parameter
        try:
            strength = max(0, min(100, int(request.form.get('strength', 50))))
        except ValueError:
            strength = 50
            
        # File handling
        original_path = save_uploaded_file(file)
        
        text = transcribe_audio(original_path)
        if not text:
            return jsonify({'error': 'Transcription failed'}), 500
            
        # Speech synthesis
        session_id = uuid.uuid4()
        neutralized_path = os.path.join(PROCESSED_FOLDER, f"{session_id}_processed.wav")
        
        accent = request.form.get('accent', 'neutral')
        tts = gTTS(text=text, lang=ACCENT_CODES.get(accent, 'en'))
        tts.save(neutralized_path)
        
        # Audio blending
        final_path = os.path.join(PROCESSED_FOLDER, f"{session_id}_final.wav")
        blend_audio(original_path, neutralized_path, strength, final_path)
        
        return send_file(
            neutralized_path, 
            mimetype='audio/wav',
            as_attachment=True,
            download_name='processed_audio.wav'
        )
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Cleanup temporary files but keep final output until cleanup thread removes it
        for path in [original_path, neutralized_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {path}: {str(e)}")

if __name__ == '__main__':
    # Start background cleanup thread
    cleanup_thread = threading.Thread(target=scheduled_cleanup, daemon=True)
    cleanup_thread.start()
    
    # Initial cleanup
    cleanup_old_files()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

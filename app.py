#!/usr/bin/env python3
"""
Flask web application for Musical Melody of Emotion
Demonstrates Korean text emotion analysis and EEG-based emotion detection
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from integrated_emotion_engine import IntegratedEmotionEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion-music-app'

# Initialize the emotion engine
emotion_engine = IntegratedEmotionEngine()

@app.route('/')
def index():
    """Main page with emotion analysis interface"""
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze Korean text for emotion and generate music recommendations"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Analyze the Korean text
        result = emotion_engine.analyze_text(text)
        
        return jsonify({
            'success': True,
            'result': result,
            'input_text': text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_eeg', methods=['POST'])
def analyze_eeg():
    """Analyze synthetic EEG data for emotion detection"""
    try:
        data = request.get_json()
        signal_type = data.get('signal_type', 'joy')
        duration = float(data.get('duration', 5.0))
        
        # Generate synthetic EEG data based on signal type
        fs = 512
        t = np.arange(0, duration, 1.0 / fs)
        
        if signal_type == 'joy':
            # Alpha and beta frequencies for joy
            signal = 0.8 * np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        elif signal_type == 'sadness':
            # Theta frequencies for sadness
            signal = 1.0 * np.sin(2 * np.pi * 6 * t) + 0.2 * np.sin(2 * np.pi * 10 * t)
        elif signal_type == 'anger':
            # Beta and gamma frequencies for anger
            signal = 1.0 * np.sin(2 * np.pi * 20 * t) + 0.8 * np.sin(2 * np.pi * 40 * t)
        else:
            # Neutral mixed signal
            signal = 0.5 * np.sin(2 * np.pi * 8 * t) + 0.3 * np.sin(2 * np.pi * 15 * t)
        
        # Add some noise
        signal += 0.05 * np.random.randn(t.size)
        
        # Create multi-channel EEG (simulate 2 channels)
        eeg_data = np.stack([signal, signal * 0.9])
        
        # Analyze the EEG data
        result = emotion_engine.analyze_eeg(eeg_data, sampling_rate=fs)
        
        return jsonify({
            'success': True,
            'result': result,
            'signal_type': signal_type,
            'duration': duration
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Emotion Analysis Engine is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
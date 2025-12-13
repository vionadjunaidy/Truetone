from flask import Flask, request, jsonify
from flask_cors import CORS
from model_inference import EmotionModel
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React app

# Initialize model
model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model_with_gender.pt')
emotion_model = EmotionModel(model_path)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Emotion detection API is running'})

@app.route('/api/analyze', methods=['POST'])
def analyze_emotion():
    try:
        data = request.json
        text = data.get('text', '')
        gender = data.get('gender', '')
        
        if not text or not gender:
            return jsonify({
                'error': 'Both text and gender are required'
            }), 400
        
        # Run inference
        result = emotion_model.predict(text, gender)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print(f"Loading model from: {model_path}")
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)

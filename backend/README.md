# Backend API for Emotion Detection

This backend serves the PyTorch emotion detection model via a REST API.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the model file `best_model_with_gender.pt` is in the root directory (one level up from the backend folder).

3. Start the Flask server:
```bash
cd backend
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/health`
- Returns API status

### Analyze Emotion
- **POST** `/api/analyze`
- Request body:
  ```json
  {
    "text": "Your text to analyze",
    "gender": "male" or "female"
  }
  ```
- Response:
  ```json
  {
    "label": "Emotion label",
    "confidence": 0.85,
    "cues": ["cue1", "cue2", "cue3"]
  }
  ```

## Notes

- The model inference code (`model_inference.py`) may need adjustments based on your specific model architecture
- If your model uses a different tokenizer or preprocessing, update the `_preprocess_text` and tokenization logic
- The emotion labels may need to be adjusted to match your model's output classes

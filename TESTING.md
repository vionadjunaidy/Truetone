# Testing Guide

This guide will help you test the emotion detection integration.

## Prerequisites

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Node.js dependencies (if not already done):**
   ```bash
   npm install
   ```

3. **Verify the model file exists:**
   - Make sure `best_model_with_gender.pt` is in the root directory

## Testing Methods

### Method 1: Automated Test Script (Recommended)

1. **Start the backend server** (Terminal 1):
   ```bash
   cd backend
   python app.py
   ```
   You should see:
   ```
   Loading model from: [path]
   Starting Flask server on http://localhost:5000
   * Running on http://127.0.0.1:5000
   ```

2. **Run the test script** (Terminal 2):
   ```bash
   cd backend
   python test_api.py
   ```
   
   This will test:
   - Health endpoint
   - Analyze endpoint with multiple test cases

### Method 2: Manual API Testing with curl

1. **Start the backend server:**
   ```bash
   cd backend
   python app.py
   ```

2. **Test health endpoint:**
   ```bash
   curl http://localhost:5000/health
   ```
   Expected response:
   ```json
   {"status": "healthy", "message": "Emotion detection API is running"}
   ```

3. **Test analyze endpoint:**
   ```bash
   curl -X POST http://localhost:5000/api/analyze \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"I am feeling happy today!\", \"gender\": \"female\"}"
   ```
   
   Expected response:
   ```json
   {
     "label": "Happy",
     "confidence": 0.85,
     "cues": ["Positive wording", "Upbeat tone", "Enthusiastic language"]
   }
   ```

### Method 3: Full Integration Test (Frontend + Backend)

1. **Start the backend** (Terminal 1):
   ```bash
   cd backend
   python app.py
   ```

2. **Start the React frontend** (Terminal 2):
   ```bash
   npm run dev
   ```

3. **Open your browser:**
   - Navigate to the URL shown in the terminal (usually `http://localhost:5173`)

4. **Test the UI:**
   - Enter some text in the text area (e.g., "I'm feeling great today!")
   - Select a gender (Male or Female)
   - Click the "Analyze" button
   - You should see the emotion result with confidence and cues

## Troubleshooting

### Backend won't start

**Error: Module not found**
- Make sure you've installed all dependencies: `pip install -r requirements.txt`

**Error: Model file not found**
- Verify `best_model_with_gender.pt` is in the root directory (not in the backend folder)

**Error: Port 5000 already in use**
- Change the port in `backend/app.py`:
  ```python
  app.run(debug=True, port=5001)  # Use a different port
  ```
- Update the React app to use the new port in `src/App.jsx`:
  ```javascript
  const response = await fetch('http://localhost:5001/api/analyze', {
  ```

### Frontend can't connect to backend

**CORS errors in browser console**
- Make sure Flask-CORS is installed and the backend is running
- Check that the backend URL in `App.jsx` matches the port the backend is using

**Network error / Connection refused**
- Verify the backend is running on the correct port
- Check that both servers are running simultaneously
- Try accessing `http://localhost:5000/health` directly in your browser

### Model inference errors

**Error loading model**
- The model architecture might be different than expected
- Check the error message in the backend terminal
- You may need to adjust `backend/model_inference.py` to match your model's architecture

**Wrong emotion labels**
- Update the `emotion_labels` list in `backend/model_inference.py` to match your model's output classes

## Expected Behavior

✅ **Successful test:**
- Backend starts without errors
- Health endpoint returns 200 status
- Analyze endpoint returns emotion predictions
- Frontend displays results correctly

❌ **If something fails:**
- Check the terminal output for error messages
- Verify all dependencies are installed
- Ensure the model file is accessible
- Check that ports aren't conflicting

## Next Steps

Once testing is successful:
- Customize emotion labels to match your model
- Adjust preprocessing if needed
- Fine-tune the UI based on your results
- Add error handling improvements if needed

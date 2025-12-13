# Quick Testing Guide for localhost:5173

## How to Test if Your App is Working with the Model

### Method 1: Test in Browser (Easiest)

1. **Open your browser's Developer Console:**
   - Press `F12` or `Ctrl+Shift+I` (Windows) / `Cmd+Option+I` (Mac)
   - Go to the "Console" tab

2. **Check if backend is reachable:**
   ```javascript
   fetch('http://localhost:5000/health')
     .then(r => r.json())
     .then(console.log)
   ```
   ✅ **Expected:** `{status: "healthy", message: "Emotion detection API is running"}`

3. **Test the model directly:**
   ```javascript
   fetch('http://localhost:5000/api/analyze', {
     method: 'POST',
     headers: {'Content-Type': 'application/json'},
     body: JSON.stringify({text: "I'm feeling great!", gender: "female"})
   })
   .then(r => r.json())
   .then(console.log)
   ```
   ✅ **Expected:** `{label: "...", confidence: 0.XX, cues: [...]}`
   ❌ **If you see:** `{label: "Error", ...}` - the model needs fixing

### Method 2: Test in the UI

1. **Make sure backend is running:**
   - Check terminal where you ran `python backend/app.py`
   - Should see: "Starting Flask server on http://localhost:5000"

2. **In the app (localhost:5173):**
   - Enter some text: "I'm feeling happy today!"
   - Select a gender (Male or Female)
   - Click "Analyze"

3. **Check the results:**
   - ✅ **Working:** Shows emotion label, confidence %, and cues
   - ❌ **Not working:** Shows "Error" label or falls back to mock data

### Method 3: Check Network Tab

1. Open Developer Tools (`F12`)
2. Go to "Network" tab
3. Click "Analyze" in the app
4. Look for request to `http://localhost:5000/api/analyze`
   - ✅ **200 status:** Backend is responding
   - ✅ **Response has label/confidence:** Model is working
   - ❌ **500 status or error in response:** Model issue

### Signs the Model is Working:

✅ **Good signs:**
- Emotion label appears (not "Error")
- Confidence is a number between 0 and 1 (or 0-100%)
- Cues list appears with descriptive text
- Different emotions for different inputs

❌ **Bad signs:**
- Label shows "Error"
- Confidence is 0.0
- Cues show error messages like "Model error: ..."
- Always returns the same result regardless of input

### Quick Fix if Model Shows Errors:

If you see model errors, restart the backend:
1. Stop the backend (Ctrl+C in the terminal)
2. Restart: `cd backend && python app.py`
3. Try testing again

### Test Cases to Try:

1. **Happy text:** "I'm so excited about this project!"
2. **Sad text:** "I feel really down today."
3. **Angry text:** "This is so frustrating!"
4. **Calm text:** "Everything is peaceful and quiet."

Each should give different emotion labels if the model is working correctly.

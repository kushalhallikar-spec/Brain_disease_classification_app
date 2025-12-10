from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import base64
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# -------------------- Configuration -------------------- #
IMAGE_SIZE = 150
MODEL_PATH = "model/Brain_Disease_Classification_Using_VGG16_Model.h5"
CLASS_NAMES = ['Alzheimer','Brain_Tumor', 'Healthy', 'MS' ]
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# -------------------- Configure Gemini AI -------------------- #
# Replace with your actual API key or set as environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCXP3gztyXK0sgerGdi6oRuZGUh4TvIvi4')

try:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # List available models for debugging
    print("Checking available models...")
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            print(f"  - {m.name}")
    
    # Try different model names in order of preference
    model_names = [
        'models/gemini-1.5-flash',
        'models/gemini-1.5-pro', 
        'models/gemini-pro',
        'models/gemini-1.5-flash-latest',
        'gemini-1.5-flash',
        'gemini-pro'
    ]
    
    gemini_model = None
    for model_name in model_names:
        try:
            gemini_model = genai.GenerativeModel(model_name)
            # Test the model with a simple prompt
            test_response = gemini_model.generate_content("Hi")
            print(f"✓ Gemini AI initialized successfully with model: {model_name}")
            break
        except Exception as model_error:
            print(f"  Model '{model_name}' failed: {model_error}")
            continue
    
    if gemini_model is None:
        print("✗ No working Gemini model found")
        print(f"Available models: {available_models}")
        
except Exception as e:
    print(f"✗ Gemini AI initialization failed: {e}")
    print("  Chatbot will not be available. Set GEMINI_API_KEY to enable it.")
    gemini_model = None

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -------------------- Load Model -------------------- #
print("Loading VGG16 model...")
try:
    model = load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# -------------------- System Context for Chatbot -------------------- #
CHATBOT_CONTEXT = """You are a helpful medical AI assistant specializing in brain diseases. 
You help users understand brain conditions including Brain Tumors, Alzheimer's disease, Multiple Sclerosis (MS), and general brain health.

Key information:
- Brain Tumor: Abnormal growth of cells in the brain, can be benign or malignant
- Alzheimer's: Progressive neurodegenerative disease causing memory loss and cognitive decline
- Multiple Sclerosis (MS): Autoimmune disease affecting the central nervous system
- The system uses a VGG16 deep learning model for classification

Important guidelines:
- Provide clear, empathetic, and accurate information
- Always remind users that AI predictions are not a substitute for professional medical diagnosis
- Encourage users to consult healthcare professionals for serious concerns
- Be supportive and understanding when discussing sensitive health topics
- Keep responses concise but informative (2-3 paragraphs maximum)
"""

# -------------------- Helper Functions -------------------- #
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess image for prediction"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert grayscale to RGB if needed
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Normalize
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# -------------------- Routes -------------------- #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'chatbot_available': gemini_model is not None,
        'gemini_model_type': str(type(gemini_model)) if gemini_model else None
    })

@app.route('/test-chat', methods=['GET'])
def test_chat():
    """Test chatbot with a simple message"""
    if gemini_model is None:
        return jsonify({
            'error': 'Gemini model is None',
            'success': False
        })
    
    try:
        response = gemini_model.generate_content("Say 'Hello' in one word")
        return jsonify({
            'success': True,
            'response': response.text,
            'model_type': str(type(gemini_model))
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'success': False
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict brain disease from uploaded image"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg, bmp, tiff'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_img = preprocess_image(filepath)
        
        if processed_img is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        # Get all class probabilities
        all_predictions = {
            CLASS_NAMES[i]: float(predictions[0][i]) * 100 
            for i in range(len(CLASS_NAMES))
        }
        
        # Read image and convert to base64 for display
        with open(filepath, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'all_predictions': {k: round(v, 2) for k, v in all_predictions.items()},
            'image': img_base64
        })
        
    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of available classes"""
    return jsonify({
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES)
    })

@app.route('/chat', methods=['POST'])
def chat():
    """AI Chatbot endpoint using Google Gemini"""
    print("\n=== CHAT REQUEST RECEIVED ===")
    
    if gemini_model is None:
        error_msg = 'AI Chatbot not available. Please configure GEMINI_API_KEY environment variable.'
        print(f"ERROR: {error_msg}")
        return jsonify({
            'error': error_msg,
            'success': False
        }), 500
    
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        print(f"User message: {user_message}")
        
        if not user_message:
            return jsonify({'error': 'No message provided', 'success': False}), 400
        
        # Create prompt with context
        full_prompt = f"{CHATBOT_CONTEXT}\n\nUser: {user_message}\n\nAssistant:"
        
        print("Sending request to Gemini...")
        
        # Use the GLOBAL gemini_model that was already initialized
        response = gemini_model.generate_content(full_prompt)
        
        print(f"Gemini response received: {response.text[:100]}...")
        
        return jsonify({
            'success': True,
            'response': response.text
        })
        
    except Exception as e:
        error_details = f"Chat error: {type(e).__name__}: {str(e)}"
        print(f"\n!!! ERROR DETAILS !!!")
        print(error_details)
        print(f"!!! END ERROR !!!\n")
        
        # Return more detailed error to frontend
        return jsonify({
            'error': f'Chat failed: {str(e)}',
            'error_type': type(e).__name__,
            'success': False
        }), 500

# -------------------- Run Server -------------------- #
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🧠 Brain Disease Classification Server")
    print("="*50)
    print(f"Model Status: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print(f"Chatbot Status: {'✓ Available' if gemini_model else '✗ Not Available (Set GEMINI_API_KEY)'}")
    print("="*50)
    print("Server starting at http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
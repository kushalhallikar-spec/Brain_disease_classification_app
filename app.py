from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import base64

import os
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

# -------------------- Flask App -------------------- #
app = Flask(__name__)
CORS(app)

# -------------------- Configuration -------------------- #
IMAGE_SIZE = 150
MODEL_PATH = "model/Brain_Disease_Classification_Using_VGG16_Model.h5"
CLASS_NAMES = ['Alzheimer', 'Brain_Tumor', 'Healthy','Invalid', 'MS']
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# -------------------- OpenAI Chatbot -------------------- #
openai_client = None

try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("✓ OpenAI chatbot connected")
except Exception as e:
    print("✗ OpenAI chatbot failed:", e)
    openai_client = None

CHATBOT_CONTEXT = """
You are a professional medical AI assistant specializing in brain diseases.

You can explain:
- Brain Tumors
- Alzheimer’s disease
- Multiple Sclerosis (MS)
- MRI-based diagnosis concepts

Rules:
- Do NOT give a medical diagnosis
- Be clear, calm, and professional
- Always advise consulting a medical professional
- Keep answers concise and human-like
"""

# -------------------- Load Model -------------------- #
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print("✗ Model loading failed:", e)
    model = None

# -------------------- Create Upload Folder -------------------- #
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -------------------- Helper Functions -------------------- #
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------- Routes -------------------- #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    processed_img = preprocess_image(filepath)

    if processed_img is None:
        os.remove(filepath)
        return jsonify({'error': 'Invalid image'}), 400

    predictions = model.predict(processed_img, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    all_predictions = {
        CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    with open(filepath, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    os.remove(filepath)

    return jsonify({
        'success': True,
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2),
        'all_predictions': all_predictions,
        'image': image_base64
    })

@app.route('/chat', methods=['POST'])
def chat():
    if openai_client is None:
        return jsonify({
            "success": True,
            "response": "AI chatbot is temporarily unavailable. Please consult a medical professional."
        })

    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"success": False, "response": "Please enter a valid question."})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CHATBOT_CONTEXT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.4,
            max_tokens=300
        )

        return jsonify({
            "success": True,
            "response": response.choices[0].message.content
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "response": f"Chatbot error: {str(e)}"
        })

# -------------------- Run Server -------------------- #
if __name__ == '__main__':
    print("\n🧠 Brain Disease Classification System")
    print("🌐 Server running at http://localhost:5000\n")
    app.run(debug=True)

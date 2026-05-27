# 🧠 Brain Disease Classification App

A deep learning web app for real-time brain disease detection from MRI scans using a fine-tuned ResNet model — deployed via Flask for real-time diagnosis support.

## 🧠 Overview
Early detection of brain diseases is critical for effective treatment outcomes. This project leverages Transfer Learning with ResNet to classify brain MRI scans into multiple disease categories, providing a fast and accessible screening tool via a web interface.

## 📊 Dataset
- **Type:** Brain MRI / CT scan images
- **Problem:** Multi-class classification
- **Classes:** Tumor / Alzheimer's / Multiple Sclerosis / Normal
- **Preprocessing:** Resized to uniform shape, normalised pixel values, train/validation split

## ⚙️ ML Pipeline
```
MRI Images
   │
   ▼
Preprocessing (Resize, Normalize)
   │
   ▼
Data Augmentation (Rotation, Flip, Zoom)
   │
   ▼
ResNet (Fine-Tuned via Transfer Learning)
   │
   ▼
Softmax Output → Disease Class
   │
   ▼
Flask Web App (Real-time Prediction)
```

## 🤖 Model Architecture
- **Base Model:** ResNet (pre-trained on ImageNet)
- **Fine-tuning:** Top layers retrained on brain MRI dataset
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Augmentation:** Rotation, horizontal flip, zoom

## 🛠️ Tech Stack
- Python, TensorFlow, Keras
- ResNet (Transfer Learning)
- OpenCV, NumPy, Matplotlib
- Flask (Web App), HTML/CSS (Templates)
- OpenAI GPT (AI Chatbot for Q&A)

## 🚀 Run Locally
```bash
git clone https://github.com/kushalhallikar-spec/Brain_disease_classification_app.git
cd Brain_disease_classification/app
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:5000` in your browser, upload an MRI image, and get a prediction.

## 📁 Project Structure
```
Brain_disease_classification_app/
│
├── app.py                                   # Flask app — routes & prediction logic
├── Total_Disease_Brain_ResNet_FineTuned.py  # Model training script
├── requirements.txt                         # Python dependencies
├── .python-version                          # Python 3.10
├── templates/                               # HTML frontend
└── static/                                  # CSS & JS
```

## 🔍 Key Insights
- Transfer learning drastically reduces training time vs training from scratch
- Data augmentation is critical for medical imaging — real datasets are small
- ResNet's residual connections handle vanishing gradients well for deep networks

## 🔮 Future Improvements
- Add Grad-CAM visualisation to highlight disease regions on MRI
- Expand to more disease classes
- Deploy on Hugging Face Spaces or Streamlit Cloud
- Improve dataset diversity for better generalisation

## 👨‍💻 Author
**Kushal Hallikar** — Aspiring Machine Learning Engineer  
[LinkedIn](https://linkedin.com/in/kushalhallikar/) | [GitHub](https://github.com/kushalhallikar-spec)

⭐ If you found this useful, consider giving it a star!

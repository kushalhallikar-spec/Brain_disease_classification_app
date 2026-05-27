<div align="center">

# 🧠 Brain Disease Classification App

### Real-time brain MRI analysis powered by Deep Learning & AI

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT-412991?style=for-the-badge&logo=openai)

</div>

---

## ✨ What is this?

> Upload a brain MRI scan → Get instant AI-powered disease classification → Ask the chatbot anything about your results.

This app uses a **fine-tuned ResNet model** trained on brain MRI scans to detect:

| Disease | Status |
|---------|--------|
| 🔴 Brain Tumor | ✅ Supported |
| 🟡 Alzheimer's Disease | ✅ Supported |
| 🟠 Multiple Sclerosis | ✅ Supported |
| 🟢 Normal | ✅ Supported |

---

## ⚙️ How It Works

```
📤 Upload MRI Scan
       │
       ▼
🔄 Preprocessing (Resize · Normalize · Augment)
       │
       ▼
🧠 ResNet (Fine-Tuned on ImageNet + Brain MRI Dataset)
       │
       ▼
📊 Softmax → Disease Class + Confidence Score
       │
       ▼
🤖 OpenAI Chatbot → Ask questions about your result
```

---

## 🛠️ Tech Stack

```
🐍 Python 3.10        🔥 TensorFlow & Keras
🧠 ResNet (TL)        👁️ OpenCV
🌐 Flask              🎨 HTML · CSS · JavaScript  
🤖 OpenAI GPT         📊 NumPy · Matplotlib
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/kushalhallikar-spec/Brain_disease_classification_app.git
cd Brain_disease_classification_app

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Run the app
python app.py
```

Open `http://localhost:5000` and upload an MRI scan.

---

## 📁 Project Structure

```
Brain_disease_classification_app/
│
├── 🐍 app.py                                   # Flask backend
├── 🧠 Total_Disease_Brain_ResNet_FineTuned.py  # Model training
├── 📋 requirements.txt                         # Dependencies
├── 🎨 templates/                               # HTML frontend
└── ⚡ static/                                  # CSS & JS
```

---

## 🔍 Key Insights

- ⚡ **Transfer Learning** cuts training time drastically vs from scratch
- 🔄 **Data Augmentation** is essential — medical datasets are small
- 🏗️ **ResNet residual connections** solve vanishing gradient for deep networks

---

## 🔮 Roadmap

- [ ] Grad-CAM visualisation to highlight disease regions
- [ ] Expand dataset diversity for better generalisation
- [ ] Deploy on Hugging Face Spaces
- [ ] Add patient report generation (PDF export)

---

<div align="center">

## 👨‍💻 Author

**Kushal Hallikar**
*Aspiring Machine Learning Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/kushalhallikar/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/kushalhallikar-spec)

⭐ **Star this repo if you found it useful!**

</div>

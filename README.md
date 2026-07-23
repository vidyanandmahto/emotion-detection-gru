<div align="center">

# 🧠 EmotionLens — Text Emotion Detection

**Deep learning NLP system that classifies free-form English text into six emotions — joy, sadness, anger, fear, love, and surprise — served through a real-time Flask web app.**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📌 Project Overview

EmotionLens is an end-to-end NLP project that takes a raw sentence (a tweet, a diary entry, a review, anything) and predicts the emotion behind it using a recurrent neural network trained from scratch on the open **Emotion** dataset (~20,000 labeled sentences).

The project covers the full machine learning lifecycle: data exploration → text preprocessing → tokenization → model design & comparison (SimpleRNN vs LSTM vs GRU) → evaluation → deployment as a polished, responsive Flask web app.

---

## ✨ Features

- 🔍 Real-time emotion prediction via AJAX (no page reload)
- 🎯 Confidence score with an animated circular meter
- 📊 Full probability breakdown across all six emotions
- 😄 Emotion-specific emoji and color coding
- 🕘 Local prediction history panel
- ⚡ One-click example sentences
- 🛡️ Input validation with empty-input warnings
- 🌌 Dark-blue glassmorphism UI with animated gradient background
- 📱 Fully responsive, mobile-friendly design
- 🧯 Non-JS fallback route (`/predict-form`) for progressive enhancement
- 🩺 `/health` endpoint for uptime checks on deployment platforms

---

## 🗂️ Dataset

The **Emotion Dataset** (`train.txt`, `val.txt`, `test.txt`) — 20,000 short English sentences, each on its own line as `sentence;emotion`.

| Split | Rows |
|---|---|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

**Classes:** `anger`, `fear`, `joy`, `love`, `sadness`, `surprise`

Class distribution is imbalanced (joy and sadness dominate; surprise is rare) — see `notebook/assets/class_distribution.png`.

---

## 🧬 Model Architecture

Three sequence models were built and trained under identical conditions, then compared on held-out test data:

```
Input → Embedding(vocab=12000, dim=64) → [SimpleRNN | LSTM | GRU](64) → Dropout(0.5)
      → Dense(32, relu) → Dropout(0.3) → Dense(6, softmax)
```

Trained with the Adam optimizer, categorical cross-entropy loss, and `EarlyStopping` on validation loss.

### 📈 Results

| Model | Test Accuracy |
|---|---|
| SimpleRNN | 77.2% |
| LSTM | 87.3% |
| **GRU (best)** | **89.6%** |

**Best model — GRU — classification report:**

```
              precision    recall  f1-score   support

       anger       0.89      0.93      0.91       542
        fear       0.84      0.88      0.86       475
         joy       0.92      0.94      0.93      1352
        love       0.75      0.76      0.75       328
     sadness       0.96      0.94      0.95      1159
    surprise       0.68      0.40      0.50       144

    accuracy                           0.90      4000
   macro avg       0.84      0.81      0.82      4000
weighted avg       0.89      0.90      0.89      4000
```

> `surprise` is the rarest class in the dataset and is the model's weakest category — a natural target for future improvement (e.g. class weighting or oversampling).

All training curves, confusion matrices, and comparison charts are saved in `notebook/assets/`.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Deep Learning | TensorFlow / Keras |
| Classical ML / Preprocessing | scikit-learn, NLTK |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Backend | Flask |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Deployment | Render / Railway / PythonAnywhere, gunicorn |

---

## 📁 Folder Structure

```
Emotion-Detection/
│
├── app.py                     # Flask application
├── train_pipeline.py          # Full training pipeline (Phases 1–10)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
│
├── model/
│   ├── emotion_model.keras    # Trained GRU model
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── max_len.pkl
│   └── model_info.json
│
├── notebook/
│   ├── Emotion_Detection.ipynb
│   └── assets/                # Generated charts & reports
│
├── templates/
│   ├── index.html
│   ├── about.html
│   └── result.html
│
└── static/
    ├── css/style.css
    ├── js/script.js
    └── logo.png
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Emotion-Detection.git
cd Emotion-Detection

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (First run only) download NLTK corpora
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 🚀 How to Run

### Train the model (optional — a trained model is already included in `model/`)

```bash
python3 train_pipeline.py
```

### Run the web app

```bash
python3 app.py
```

Then open **http://localhost:5000** in your browser.

### Production server

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## 🖼️ Screenshots

|---|---|---|
| `Web Images\Screenshot (162).png` |

---

## 🔮 Future Improvements

- Address class imbalance for `surprise` (oversampling / class weights / focal loss)
- Fine-tune a pretrained transformer (e.g. DistilBERT) for higher accuracy
- Add multi-label emotion detection (a sentence can carry more than one emotion)
- Add a REST API key / rate limiting for public deployment
- Add automated tests (pytest) and CI (GitHub Actions)
- Dockerize the application

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Vidyanand Mahto

Built as a complete, portfolio-ready NLP + Flask deep learning project.
Feel free to fork, star ⭐, and adapt it for your own dataset.

# 📰 Fake News Detection System

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application that classifies news articles as **True** or **Fake** using Natural Language Processing (NLP) and supervised learning. Built with Python, Scikit-Learn, and Streamlit, this system provides real-time predictions along with statistical confidence metrics.

---

## 📌 Overview

In the modern digital era, the propagation of misinformation and fake news is a critical challenge. This project provides an accessible, automated solution to verify the authenticity of news articles. 

By leveraging **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization** and a pre-trained classification model, the application processes raw text, extracts key linguistic features, and predicts whether an article is factual or fabricated with a calculated confidence percentage.

### Key Value Propositions
* **Real-time Inference:** Paste any news article and get classification results in seconds.
* **Explainable Confidence:** Displays a probability score indicating the model's certainty.
* **Robust Preprocessing:** Cleans raw text by removing HTML tags, URLs, punctuation, digits, and noise before classification.

---

## ✨ Features

- [x] **Advanced Text Preprocessing:** Custom pipeline to strip URLs, HTML tags, punctuation, and non-alphabetic characters.
- [x] **TF-IDF Feature Extraction:** Converts unstructured text into highly informative numerical feature vectors.
- [x] **Interactive Web UI:** Clean, responsive, and minimalist interface built with Streamlit.
- [x] **Confidence Metrics:** Visual progress bars displaying prediction probability.
- [x] **Model Training Notebook:** Includes the complete exploratory data analysis (EDA) and training pipeline.

---

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **Frontend UI:** Streamlit
* **Machine Learning:** Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Natural Language Processing:** NLTK, Regular Expressions (`re`)
* **Model Serialization:** Joblib

---

## 📐 System Architecture

The application follows a modular architecture separating data processing, model storage, and the user interface.

```
┌────────────────┐      ┌─────────────────────┐      ┌──────────────────────┐
│  User Input    │ ───> │  Text Preprocessing │ ───> │ TF-IDF Vectorization │
│  (Raw Article) │      │  (Regex & Cleaning) │      │  (vectorizer.pkl)    │
└────────────────┘      └─────────────────────┘      └──────────┬───────────┘
                                                                │
                                                                ▼
┌────────────────┐      ┌─────────────────────┐      ┌──────────────────────┐
│ Streamlit UI   │ <─── │ Prediction Output   │ <─── │ Classification Model │
│ (Render Result)│      │ & Confidence Score  │      │  (model.pkl)         │
└────────────────┘      └─────────────────────┘      └──────────────────────┘
```

### Directory Structure

```
├── dataset/                  # Raw training datasets
│   ├── Fake.csv              # Fabricated news articles
│   └── True.csv              # Authentic news articles
├── models/                   # Serialized ML artifacts
│   ├── model.pkl             # Trained classification model
│   └── vectorizer.pkl        # Fitted TF-IDF Vectorizer
├── notebooks/                # Jupyter notebooks for R&D
│   └── fake_news.ipynb       # Model training and evaluation pipeline
├── src/                      # Application source code
│   └── app.py                # Streamlit web application
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

```bash
python --version
```

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/shreshthasharma110/fake_news_detection.git
   cd fake_news_detection
   ```

2. **Create a Virtual Environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration & Preprocessing

The text preprocessing pipeline is critical for removing noise from web-scraped news articles. The system processes text using the following pipeline:

```python
def preprocess_text(text):
    text = text.lower()                                             # Lowercase text
    text = re.sub(r'https?://\S+', '', text)                        # Remove URLs
    text = re.sub(r'<.*?>', '', text)                               # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', ' ', text)                                 # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)                            # Remove words containing digits
    return text
```

---

## 💻 Usage

### Running the Web Application

To launch the Streamlit interface locally, run the following command from the root directory:

```bash
streamlit run src/app.py
```

Once started, the application will be accessible in your web browser at `http://localhost:8501`.

### Step-by-Step Guide

1. Open the application in your browser.
2. Paste the full body of a news article into the text area.
3. Click the **🔍 Predict** button.
4. The system will clean the text, vectorize it, and display:
   * A **TRUE** (Success) or **FAKE** (Error) classification banner.
   * A **Confidence Metric** showing the prediction probability percentage.

---

## 📊 Model Training Pipeline

The model is trained on the standard Fake and True News dataset (e.g., ISOT Dataset). To retrain or customize the model:

1. Navigate to the `notebooks/` directory.
2. Open `fake_news.ipynb` using Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook notebooks/fake_news.ipynb
   ```
3. Run the cells to load the datasets from `dataset/`, preprocess the text, train the classifier (e.g., Logistic Regression, Passive Aggressive Classifier, or Naive Bayes), and export the updated `model.pkl` and `vectorizer.pkl` to the `models/` directory.

---

## 🛠️ Troubleshooting

### Common Issues

* **File Not Found Error (`model.pkl` or `vectorizer.pkl`):**
  Ensure that you are running the Streamlit app from the root directory of the project, or that the relative paths in `src/app.py` resolve correctly to the `models/` directory.
  
* **Memory Issues during Training:**
  If training on large datasets causes memory exhaustion, consider increasing the `max_features` parameter in the `TfidfVectorizer` constructor within the notebook.

* **NLTK Download Issues:**
  If NLTK dependencies fail to load, run a python shell and download the required packages manually:
  ```python
  import nltk
  nltk.download('stopwords')
  ```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the model accuracy, add new features, or enhance the UI, please follow these steps:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## ✉️ Contact & Acknowledgments

* **Developer:** [Shreshtha Sharma](https://github.com/shreshthasharma110)
* **Dataset Source:** [ISOT Fake News Dataset](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php)
* Special thanks to the open-source community for providing the tools and libraries that made this project possible.
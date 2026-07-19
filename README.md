# 🧠 Emotion Detection & Learning Support Engine

An AI-powered web application that detects students' emotions from text using deep learning models and provides personalized learning guidance using Google's Gemini API. The application features an intuitive Streamlit interface, prediction history management, and interactive analytics dashboards to support emotion-aware learning.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Author](#author)
- [Repository](#repository)

---

## 📖 Overview

The Emotion Detection & Learning Support Engine is designed to identify the emotional state of students from textual input. It leverages deep learning techniques (BERT and BiLSTM) to classify emotions and uses Google Gemini API to generate personalized learning guidance.

The application helps improve student engagement by offering emotion-aware suggestions and visualizing historical trends through an analytics dashboard.

---

## 🎥 Demo Video

[Click here to view the project demo video on Google Drive](https://drive.google.com/drive/folders/10pOYXbRie0qgLF93Dt4tBgL0kbA14F1q)

---

## ✨ Features

- 😊 Emotion detection from student text
- 🤖 Deep Learning using BERT and BiLSTM
- 💡 AI-generated personalized learning guidance using Google Gemini
- 📊 Interactive analytics dashboard
- 📈 Confidence score visualization
- 💾 Prediction history stored in SQLite
- 📄 CSV backup of records
- 🔐 Session management
- 🎨 User-friendly Streamlit interface
- ⚡ Fast and lightweight deployment

---

## 🛠 Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Frontend | Streamlit |
| Deep Learning | BERT, BiLSTM |
| ML Framework | PyTorch, Transformers |
| AI Integration | Google Gemini API |
| Database | SQLite |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Version Control | Git & GitHub |

---

## 🏗 Project Architecture

```
Student Input
      │
      ▼
Text Preprocessing
      │
      ▼
Emotion Detection Model
(BERT / BiLSTM)
      │
      ▼
Detected Emotion
      │
      ▼
Gemini API
(Personalized Guidance)
      │
      ▼
Store Results
(SQLite + CSV)
      │
      ▼
Analytics Dashboard
```

---

## 📂 Project Structure

```
EduEmotion-AI/
│
├── data/
│   ├── emotion_records_backup.csv
│   ├── generate_dataset.py
│   └── student_emotions_dataset.csv
│
├── models/
│   ├── bert/
│   └── bilstm/
│
├── notebooks/
│
├── src/
│   ├── analytics/
│   ├── auth/
│   ├── database/
│   ├── services/
│   └── utils/
│
├── tests/
├── training/
│
├── app.py
├── config.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔄 Workflow

1. User enters text describing their current feelings.
2. Input text is preprocessed.
3. BERT/BiLSTM predicts the emotion.
4. Gemini API generates personalized learning suggestions.
5. Results are stored in SQLite and CSV.
6. Dashboard visualizes emotion trends and statistics.

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/Nisanthpakanati/EduEmotion-AI.git
```

```bash
cd EduEmotion-AI
```

### Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

---

## ▶ Running the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser at

```
http://localhost:8501
```

---

## 📊 Outputs

The application provides:

- Emotion Prediction
- Confidence Score
- Personalized Learning Guidance
- Emotion Analytics Dashboard
- Prediction History
- CSV Backup

---

## 📸 Screenshots

Screenshots will be added after deployment.

Suggested screenshots:

- Home Page
- Emotion Detection Result
- Learning Guidance
- Analytics Dashboard
- History Page

---

## 🔮 Future Enhancements

- Voice Emotion Detection
- Facial Emotion Recognition
- Multi-language Support
- Cloud Database Integration
- User Authentication
- Export Reports to PDF
- Real-time Monitoring
- Mobile Application

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📜 License

This project is developed for educational and internship purposes.

---

## 👨‍💻 Author

**Nisanth Pakanati**

GitHub:
https://github.com/Nisanthpakanati

---

## 🌐 Repository

https://github.com/Nisanthpakanati/EduEmotion-AI

---

## ⭐ Acknowledgements

- Streamlit
- Hugging Face Transformers
- Google Gemini API
- PyTorch
- Pandas
- Plotly
- SQLite
- SRM University AP
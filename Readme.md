# Diabetes Prediction System
A comprehensive Machine Learning-powered web application for diabetes risk assessment and prediction using Support Vector Machine (SVM) algorithm.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Screenshots](#️-screenshots)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#️-disclaimer)

## 🎯 Overview
The Diabetes Prediction System is an advanced ML-powered health analytics platform that predicts diabetes risk based on various clinical parameters. Built with Streamlit, it provides an intuitive interface for healthcare professionals and individuals to assess diabetes risk factors.

### Key Highlights:
- ✅ ~77% prediction accuracy using SVM algorithm
- 📊 Interactive data visualization and analytics
- 🎨 Modern, responsive UI with professional medical theme
- 📈 Real-time risk factor assessment
- 🔬 Based on Pima Indians Diabetes Database

## ✨ Features

### 1. Diabetes Prediction
- Real-time prediction based on 8 clinical parameters
- Confidence scoring for prediction results
- Color-coded results (Positive/Negative)
- Personalized health recommendations

### 2. Risk Factor Analysis
- Automatic detection of risk factors
- Visual risk indicators
- Threshold-based warnings

### 3. Analytics Dashboard
- Comprehensive dataset statistics
- Interactive visualizations using Plotly
- Distribution analysis
- Correlation heatmaps
- Age and outcome distribution charts

### 4. Health Tips & Recommendations
- Personalized health guidelines
- Exercise recommendations
- Dietary suggestions
- Lifestyle modification tips

## 📊 Dataset
The model is trained on the **Pima Indians Diabetes Database** containing 768 samples with the following features:

| Feature | Description | Unit |
|---------|-------------|------|
| Pregnancies | Number of times pregnant | - |
| Glucose | Plasma glucose concentration | mg/dL |
| Blood Pressure | Diastolic blood pressure | mm Hg |
| Skin Thickness | Triceps skin fold thickness | mm |
| Insulin | 2-Hour serum insulin | mu U/ml |
| BMI | Body mass index | kg/m² |
| Diabetes Pedigree Function | Genetic predisposition factor | - |
| Age | Age in years | years |
| Outcome | Class variable (0: Non-diabetic, 1: Diabetic) | - |

### Dataset Statistics
- **Total Records:** 768
- **Diabetic Cases:** ~34.9%
- **Non-diabetic Cases:** ~65.1%
- **Average Age:** 33.2 years
- **Average Glucose:** 121.7 mg/dL

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/AyushMaurya13/diabetes-prediction.git
cd diabetes-prediction
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python --version
streamlit --version
```

## 💻 Usage

### Running the Application

1. **Start the Streamlit Server:**
```bash
streamlit run coap.py
```

2. **Access the Application:**
   - Open your browser and navigate to: `http://localhost:8501`
   - The application will automatically open in your default browser

### Making a Prediction

1. **Navigate to Prediction Tab**

2. **Enter Patient Information:**
   - Number of Pregnancies (0-20)
   - Glucose Level (0-200 mg/dL)
   - Blood Pressure (0-140 mm Hg)
   - Skin Thickness (0-100 mm)
   - Insulin Level (0-900 mu U/ml)
   - BMI (0-70 kg/m²)
   - Diabetes Pedigree Function (0.0-3.0)
   - Age (1-120 years)

3. **Click "Predict Diabetes Risk"**

4. **View Results:**
   - Prediction outcome (Positive/Negative)
   - Confidence score
   - Risk factors analysis
   - Health recommendations

### Exploring Analytics

1. Navigate to **Analytics Tab**
2. View:
   - Dataset overview statistics
   - Distribution charts
   - Correlation matrices
   - Age vs outcome analysis

## 🔬 Model Details

**Algorithm:** Support Vector Machine (SVM)

**Configuration:**
- Kernel: Linear
- Training Accuracy: ~78.0%
- Test Accuracy: ~77.3%

### Training Process

The model was trained using the following pipeline:

1. **Data Preprocessing:**
   - Handling missing values (zeros replaced with median)
   - Feature scaling using StandardScaler
   - Train-test split (80-20)

2. **Model Training:**
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model Training
classifier = SVC(kernel='linear')
classifier.fit(X_train_scaled, Y_train)
```

3. **Evaluation Metrics:**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

### Performance Metrics

```
Confusion Matrix:
[[91  9]
 [26 28]]

Classification Report:
              precision    recall  f1-score   support
           0       0.78      0.91      0.84       100
           1       0.76      0.52      0.62        54
    accuracy                           0.77       154
```

## 📁 Project Structure

```
diabetes-prediction/
│
├── app.py                    # Main Streamlit application
├── diabetes.csv               # Training dataset
├── diabetes_model.pkl         # Trained SVM model
├── Diabetes_Prediction.ipynb  # Jupyter notebook for model development
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
└── .gitignore                # Git ignore file
```

### File Descriptions
- **coap.py:** Main application file containing the Streamlit UI and prediction logic
- **diabetes.csv:** Pima Indians Diabetes Database
- **diabetes_model.pkl:** Pre-trained SVM model (serialized)
- **Diabetes_Prediction.ipynb:** Complete model development workflow
- **requirements.txt:** All required Python packages

## 📦 Dependencies

### Core Libraries
```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
plotly==5.17.0
```

### Additional Requirements
- Python 3.9+
- pickle (built-in)
- Standard library modules

## 🖼️ Screenshots

### Main Interface
![Main Interface](screenshots/main_interface.png)

### Prediction Results
![Prediction Results](screenshots/prediction.png)

### Analytics Dashboard
![Analytics Dashboard](screenshots/analytics.png)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**
```bash
git checkout -b feature/AmazingFeature
```

3. **Commit Your Changes**
```bash
git commit -m 'Add some AmazingFeature'
```

4. **Push to the Branch**
```bash
git push origin feature/AmazingFeature
```

5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2024 AyushMaurya13

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## ⚠️ Disclaimer

**IMPORTANT:** This application is for educational and informational purposes only.

- ❌ NOT a substitute for professional medical advice, diagnosis, or treatment
- ❌ NOT intended for clinical decision-making
- ✅ Always consult qualified healthcare professionals for medical advice
- ✅ Never rely solely on this tool for diabetes diagnosis

### Medical Advice
Any health-related information provided by this application should be verified with a licensed healthcare provider. The predictions are based on statistical models and may not account for individual circumstances.

## 📞 Contact & Support

- **Developer:** AyushMaurya13
- **Email:** your.email@example.com
- **GitHub:** [@AyushMaurya13](https://github.com/AyushMaurya13)
- **Issues:** [GitHub Issues](https://github.com/AyushMaurya13/diabetes-prediction/issues)

## 🙏 Acknowledgments

- **Dataset:** Pima Indians Diabetes Database (UCI Machine Learning Repository)
- **Framework:** Streamlit for the amazing web framework
- **ML Library:** scikit-learn for machine learning algorithms
- **Visualization:** Plotly for interactive charts

## 🔮 Future Enhancements

- [ ] Deep Learning models (Neural Networks)
- [ ] Multi-model ensemble approach
- [ ] Mobile application development
- [ ] Integration with wearable devices
- [ ] Multi-language support
- [ ] Export reports as PDF
- [ ] User authentication system
- [ ] Historical data tracking

---

<div align="center">

**Made with ❤️ for Healthcare Innovation**
**Author Ayush Kumar Maurya**
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg) ![Python](https://img.shields.io/badge/Powered%20by-Python-blue.svg) ![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)

© 2024 Diabetes Prediction System | All Rights Reserved


</div>

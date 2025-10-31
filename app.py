
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for classy design with professional medical theme - FULLY FIXED
st.markdown("""
    <style>
    /* Main background with elegant gradient */
    .main {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
    }
    
    /* Card styling with glassmorphism effect */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Header styling with modern gradient */
    .main-header {
        background: linear-gradient(135deg, #0093E9 0%, #80D0C7 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(0, 147, 233, 0.3);
    }
    
    .main-header h1 {
        font-size: 3.2rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        letter-spacing: -1px;
    }
    
    .main-header p {
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Input container with clean white background */
    .input-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    
    /* Prediction result boxes with modern colors */
    .prediction-box {
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 25px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        letter-spacing: 1px;
    }
    
    .positive {
        background: linear-gradient(135deg, #FF6B6B 0%, #EE5A6F 100%);
        color: white;
        border: 3px solid rgba(255, 255, 255, 0.3);
    }
    
    .negative {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: 3px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Metrics cards with elegant gradients */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
    }
    
    .metric-card h3, .metric-card h2 {
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Button styling with professional blue */
    .stButton>button {
        background: linear-gradient(135deg, #0093E9 0%, #80D0C7 100%);
        color: white;
        border: none;
        padding: 18px 35px;
        font-size: 1.15rem;
        font-weight: 600;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 147, 233, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 147, 233, 0.5);
        background: linear-gradient(135deg, #0081CC 0%, #6BBFB8 100%);
    }
    
    /* Info boxes - FIXED for dark background with glassmorphism */
    .info-box {
        background: rgba(255, 100, 190, 1);
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 5px solid #0093E9;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .info-box h3 {
        color: #1e3c72 !important;
        margin-bottom: 15px;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .info-box p {
        color: #2c3e50 !important;
        line-height: 1.8;
        font-size: 1.05rem;
        margin-bottom: 10px;
    }
    
    .info-box ul, .info-box ol {
        color: #2c3e50 !important;
        line-height: 2;
        font-size: 1.05rem;
        padding-left: 25px;
    }
    
    .info-box li {
        margin-bottom: 10px;
        color: #2c3e50 !important;
    }
    
    .info-box strong {
        color: #1e3c72 !important;
        font-weight: 600;
    }
    
    /* Slider styling */
    .stSlider>div>div>div>div {
        background: linear-gradient(90deg, #0093E9 0%, #80D0C7 100%);
    }
    
    /* Number input and text styling */
    .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 10px;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #0093E9;
        box-shadow: 0 0 0 0.2rem rgba(0, 147, 233, 0.25);
    }
    
    /* Main headings on dark background - WHITE */
    h1, h2, h3 {
        color: white !important;
    }
    
    /* Content area headings */
    .element-container h3 {
        color: white !important;
        margin-top: 20px;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    /* Sidebar text */
    .css-1d391kg h3 {
        color: white !important;
    }
    
    /* Warning and success boxes with glassmorphism */
    .stWarning {
        background-color: rgba(255, 243, 205, 0.95);
        border-left: 5px solid #ffc107;
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess {
        background-color: rgba(212, 237, 218, 0.95);
        border-left: 5px solid #28a745;
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background-color: rgba(248, 215, 218, 0.95);
        border-left: 5px solid #dc3545;
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stInfo {
        background-color: rgba(209, 236, 241, 0.95);
        border-left: 5px solid #0093E9;
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Markdown content styling */
    .stMarkdown {
        color: white;
    }
    
    /* Ensure all text is visible on dark background */
    p, li, span {
        color: white;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except:
        st.error("⚠️ Model file not found. Please ensure 'diabetes_model.pkl' is in the same directory.")
        return None

# Load dataset for statistics
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('diabetes.csv')
        return df
    except:
        st.warning("Dataset file not found. Using default values.")
        return None

# Header
st.markdown("""
    <div class="main-header">
        <h1>🏥 Diabetes Prediction System</h1>
        <p>Advanced ML-Powered Health Analytics Platform</p>
    </div>
""", unsafe_allow_html=True)

# Load model and data
model = load_model()
df = load_data()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Navigation")
    page = st.radio("", ["🔮 Prediction", "📈 Analytics", "ℹ️ About"])
    
    st.markdown("---")
    
    st.markdown("### 🎨 Model Information")
    st.info("""
        **Algorithm:** Support Vector Machine (SVM)
        
        **Accuracy:** ~77%
        
        **Features:** 8 Clinical Parameters
    """)
    
    st.markdown("---")
    
    st.markdown("### 👨‍⚕️ Medical Disclaimer")
    st.warning("""
        This tool is for educational purposes only. 
        Always consult healthcare professionals for medical advice.
    """)

if page == "🔮 Prediction":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter Patient Information")
        
        # Create input fields with better organization
        with st.container():
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                pregnancies = st.number_input(
                    "👶 Pregnancies",
                    min_value=0,
                    max_value=20,
                    value=0,
                    help="Number of times pregnant"
                )
                
                blood_pressure = st.slider(
                    "💓 Blood Pressure (mm Hg)",
                    min_value=0,
                    max_value=140,
                    value=70,
                    help="Diastolic blood pressure"
                )
                
                insulin = st.slider(
                    "💉 Insulin (mu U/ml)",
                    min_value=0,
                    max_value=900,
                    value=125,
                    help="2-Hour serum insulin"
                )
                
                dpf = st.number_input(
                    "🧬 Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.5,
                    step=0.01,
                    help="Genetic factor"
                )
            
            with col_b:
                glucose = st.slider(
                    "🍬 Glucose Level (mg/dL)",
                    min_value=0,
                    max_value=200,
                    value=120,
                    help="Plasma glucose concentration"
                )
                
                skin_thickness = st.slider(
                    "📏 Skin Thickness (mm)",
                    min_value=0,
                    max_value=100,
                    value=20,
                    help="Triceps skin fold thickness"
                )
                
                bmi = st.number_input(
                    "⚖️ BMI (kg/m²)",
                    min_value=0.0,
                    max_value=70.0,
                    value=25.0,
                    step=0.1,
                    help="Body Mass Index"
                )
                
                age = st.slider(
                    "🎂 Age (years)",
                    min_value=1,
                    max_value=120,
                    value=30,
                    help="Patient's age"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🔬 Predict Diabetes Risk", use_container_width=True):
            if model is not None:
                # Prepare input data
                input_data = np.array([[pregnancies, glucose, blood_pressure, 
                                       skin_thickness, insulin, bmi, dpf, age]])
                
                # Standardize the input
                scaler = StandardScaler()
                if df is not None:
                    X = df.drop('Outcome', axis=1)
                    scaler.fit(X)
                    input_data_scaled = scaler.transform(input_data)
                else:
                    input_data_scaled = input_data
                
                # Make prediction
                prediction = model.predict(input_data_scaled)
                prediction_proba = model.decision_function(input_data_scaled)
                
                # Normalize prediction probability
                confidence = min(abs(prediction_proba[0]) * 10, 100)
                
                # Display result
                if prediction[0] == 1:
                    st.markdown(f"""
                        <div class="prediction-box positive">
                            ⚠️ DIABETES POSITIVE
                            <br>
                            <small style="font-size: 1.1rem; font-weight: 400;">Confidence: {confidence:.1f}%</small>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.error("""
                        **Recommendation:** Please consult with a healthcare professional immediately 
                        for comprehensive testing and evaluation.
                    """)
                else:
                    st.markdown(f"""
                        <div class="prediction-box negative">
                            ✅ DIABETES NEGATIVE
                            <br>
                            <small style="font-size: 1.1rem; font-weight: 400;">Confidence: {confidence:.1f}%</small>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("""
                        **Recommendation:** Maintain a healthy lifestyle with regular exercise 
                        and balanced diet. Continue regular health check-ups.
                    """)
    
    with col2:
        st.markdown("### 🎯 Risk Factors")
        
        # Risk assessment based on inputs
        risk_factors = []
        
        if glucose > 140:
            risk_factors.append("🔴 High Glucose")
        if bmi > 30:
            risk_factors.append("🔴 High BMI")
        if blood_pressure > 90:
            risk_factors.append("🔴 High BP")
        if age > 45:
            risk_factors.append("🟡 Age Factor")
        if insulin > 200:
            risk_factors.append("🟡 High Insulin")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✅ No major risk factors detected")
        
        st.markdown("---")
        
        st.markdown("### 💡 Health Tips")
        st.info("""
            - 🏃 Regular exercise (30 min/day)
            - 🥗 Balanced, low-sugar diet
            - 💧 Stay hydrated
            - 😴 Adequate sleep (7-8 hours)
            - 🧘 Manage stress levels
            - 📅 Regular health screenings
        """)

elif page == "📈 Analytics":
    st.markdown("### 📊 Dataset Analytics Dashboard")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3 style="color: white;">📁 Total Records</h3>
                    <h2 style="color: white;">{}</h2>
                </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            diabetic_pct = (df['Outcome'].sum() / len(df) * 100)
            st.markdown("""
                <div class="metric-card">
                    <h3 style="color: white;">🔴 Diabetic Cases</h3>
                    <h2 style="color: white;">{:.1f}%</h2>
                </div>
            """.format(diabetic_pct), unsafe_allow_html=True)
        
        with col3:
            avg_age = df['Age'].mean()
            st.markdown("""
                <div class="metric-card">
                    <h3 style="color: white;">📅 Average Age</h3>
                    <h2 style="color: white;">{:.1f}</h2>
                </div>
            """.format(avg_age), unsafe_allow_html=True)
        
        with col4:
            avg_glucose = df['Glucose'].mean()
            st.markdown("""
                <div class="metric-card">
                    <h3 style="color: white;">🍬 Avg Glucose</h3>
                    <h2 style="color: white;">{:.1f}</h2>
                </div>
            """.format(avg_glucose), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Outcome distribution with updated colors
            fig1 = go.Figure(data=[go.Pie(
                labels=['Non-Diabetic', 'Diabetic'],
                values=df['Outcome'].value_counts().values,
                hole=0.4,
                marker=dict(colors=['#11998e', '#FF6B6B'])
            )])
            fig1.update_layout(
                title="Diabetes Distribution",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Age distribution with updated colors
            fig2 = px.histogram(
                df, 
                x='Age', 
                color='Outcome',
                nbins=20,
                title='Age Distribution by Outcome',
                color_discrete_map={0: '#11998e', 1: '#FF6B6B'}
            )
            fig2.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.9)',
                font=dict(color='white')
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔥 Feature Correlation Heatmap")
        corr = df.corr()
        fig3 = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdYlBu_r',
            title='Feature Correlation Matrix'
        )
        fig3.update_layout(
            height=600,
            font=dict(color='white')
        )
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.error("Unable to load dataset for analytics.")

else:  # About page
    st.markdown("### ℹ️ About This Application")
    
    st.markdown("""
        <div class="info-box">
            <h3>🎯 Purpose</h3>
            <p>
                This Diabetes Prediction System uses machine learning to assess the risk of diabetes 
                based on various clinical parameters. It's designed to provide quick preliminary 
                assessments and raise awareness about diabetes risk factors.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h3>🔬 Technology Stack</h3>
            <ul>
                <li><strong>Machine Learning:</strong> Support Vector Machine (SVM)</li>
                <li><strong>Framework:</strong> Streamlit</li>
                <li><strong>Data Processing:</strong> Pandas, NumPy, Scikit-learn</li>
                <li><strong>Visualization:</strong> Plotly</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h3>📋 Features Used</h3>
            <ol>
                <li><strong>Pregnancies:</strong> Number of times pregnant</li>
                <li><strong>Glucose:</strong> Plasma glucose concentration</li>
                <li><strong>Blood Pressure:</strong> Diastolic blood pressure (mm Hg)</li>
                <li><strong>Skin Thickness:</strong> Triceps skin fold thickness (mm)</li>
                <li><strong>Insulin:</strong> 2-Hour serum insulin (mu U/ml)</li>
                <li><strong>BMI:</strong> Body mass index (weight in kg/(height in m)²)</li>
                <li><strong>Diabetes Pedigree Function:</strong> Genetic predisposition factor</li>
                <li><strong>Age:</strong> Age in years</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h3>⚠️ Disclaimer</h3>
            <p>
                <strong>Important:</strong> This application is for educational and informational 
                purposes only. It should NOT be used as a substitute for professional medical advice, 
                diagnosis, or treatment. Always seek the advice of your physician or other qualified 
                health provider with any questions you may have regarding a medical condition.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h3>📧 Contact & Support</h3>
            <p>
                For questions, suggestions, or issues, please contact our development team.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p style="font-size: 1.1rem;">🏥 Diabetes Prediction System |  Machine Learning 🤖</p>
        <p style='font-size: 0.9rem; opacity: 0.8;'>
            Developed  using Streamlit | © 2025 All Rights Reserved
        </p>
    </div>
""", unsafe_allow_html=True)
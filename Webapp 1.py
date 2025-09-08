
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="Student Performance Prediction", layout="wide")

@st.cache_data
def load_model_data():
    with open('model_for_Web1.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model_data = load_model_data()
    models = model_data['models']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    model_scores = model_data['model_scores']
    categorical_columns = model_data['categorical_columns']
    data = model_data['data']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    X_train = model_data['X_train']
    y_train = model_data['y_train']
    analysis_data = model_data.get('analysis_data', {})
    feature_importance = model_data.get('feature_importance', {})
    raw_data = model_data.get('raw_data', data)
except:
    st.error("Model file not found. Please run code.ipynb first for model_for_Web1.pkl.")
    st.stop()

st.title("Student Performance Prediction Dashboard")
st.caption("A Athena Award Project, Used all light weight high accuracy model derived from my analysis if you want highest accuracy use the below link for .pkl file")
st.caption("Uses Random Data as Source File")
st.markdown("[Click Here For Source Code](https://github.com/lucks-13/student-performance)", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Performance"])

with tab1:
    st.header("Student Performance Prediction")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Personal Information")
        school = st.selectbox("School", ['GP', 'MS'])
        sex = st.selectbox("Gender", ['F', 'M'])
        age = st.slider("Age", 15, 22, 18)
        address = st.selectbox("Address", ['U', 'R'])
        famsize = st.selectbox("Family Size", ['LE3', 'GT3'])
        Pstatus = st.selectbox("Parent Status", ['T', 'A'])
        
        st.subheader("Education & Family")
        Medu = st.slider("Mother Education", 0, 4, 2)
        Fedu = st.slider("Father Education", 0, 4, 2)
        Mjob = st.selectbox("Mother Job", ['teacher', 'health', 'services', 'at_home', 'other'])
        Fjob = st.selectbox("Father Job", ['teacher', 'health', 'services', 'at_home', 'other'])
        higher = st.selectbox("Higher Education", ['yes', 'no'])
        
    with col2:
        st.subheader("School Related")
        reason = st.selectbox("Reason", ['home', 'reputation', 'course', 'other'])
        guardian = st.selectbox("Guardian", ['mother', 'father', 'other'])
        traveltime = st.slider("Travel Time", 1, 4, 2)
        studytime = st.slider("Study Time", 1, 4, 2)
        failures = st.slider("Past Failures", 0, 3, 0)
        
        st.subheader("Support & Activities")
        schoolsup = st.selectbox("School Support", ['yes', 'no'])
        famsup = st.selectbox("Family Support", ['yes', 'no'])
        paid = st.selectbox("Paid Classes", ['yes', 'no'])
        activities = st.selectbox("Activities", ['yes', 'no'])
        nursery = st.selectbox("Nursery", ['yes', 'no'])
        internet = st.selectbox("Internet", ['yes', 'no'])
        
    with col3:
        st.subheader("Social & Health")
        romantic = st.selectbox("Romantic", ['yes', 'no'])
        famrel = st.slider("Family Relations", 1, 5, 3)
        freetime = st.slider("Free Time", 1, 5, 3)
        goout = st.slider("Going Out", 1, 5, 3)
        Dalc = st.slider("Workday Alcohol", 1, 5, 1)
        Walc = st.slider("Weekend Alcohol", 1, 5, 1)
        health = st.slider("Health", 1, 5, 3)
        
        st.subheader("Academic Records")
        absences = st.slider("Absences", 0, 93, 0)
        G1 = st.slider("First Period Grade", 0, 20, 10)
        G2 = st.slider("Second Period Grade", 0, 20, 10)
        subject = st.selectbox("Subject", ['math', 'portuguese'])
    
    st.divider()
    
    st.subheader("Model Selection")
    
    st.write("**Available Models:**")
    
    col1, col2 = st.columns(2)
    selected_models = []
    
    model_list = list(models.keys())
    
    with col1:
        for i, model_name in enumerate(model_list[:len(model_list)//2 + len(model_list)%2]):
            if st.checkbox(model_name, key=f"model_{i}"):
                selected_models.append(model_name)
    
    with col2:
        for i, model_name in enumerate(model_list[len(model_list)//2 + len(model_list)%2:], start=len(model_list)//2 + len(model_list)%2):
            if st.checkbox(model_name, key=f"model_{i}"):
                selected_models.append(model_name)
    
    predict_button = st.button("Start Prediction", type="primary", use_container_width=True)
    
    if predict_button and selected_models:
        input_data = {
            'school': school, 'sex': sex, 'age': age, 'address': address,
            'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
            'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
            'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
            'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
            'activities': activities, 'nursery': nursery, 'higher': higher,
            'internet': internet, 'romantic': romantic, 'famrel': famrel,
            'freetime': freetime, 'goout': goout, 'Dalc': Dalc, 'Walc': Walc,
            'health': health, 'absences': absences, 'G1': G1, 'G2': G2,
            'subject': subject
        }
        
        input_df = pd.DataFrame([input_data])
        
        input_df['total_study_time'] = input_df['studytime'] + input_df['freetime']
        input_df['parent_education_avg'] = (input_df['Medu'] + input_df['Fedu']) / 2
        input_df['grade_trend'] = input_df['G2'] - input_df['G1']
        input_df['attendance_ratio'] = 1 - (input_df['absences'] / 93)
        input_df['support_index'] = (input_df['schoolsup'].replace({'yes': 1, 'no': 0}) +
                                    input_df['famsup'].replace({'yes': 1, 'no': 0}) +
                                    input_df['paid'].replace({'yes': 1, 'no': 0}))
        input_df['family_quality'] = input_df['famrel'] + input_df['Pstatus'].replace({'T': 2, 'A': 0})
        input_df['social_factor'] = input_df['goout'] + input_df['Dalc'] + input_df['Walc']
        input_df['motivation_score'] = (input_df['higher'].replace({'yes': 2, 'no': 0}) +
                                       input_df['internet'].replace({'yes': 1, 'no': 0}) +
                                       input_df['romantic'].replace({'yes': -1, 'no': 1}))
        
        input_df['grade_category'] = 'Medium'
        input_df['age_group'] = 'Medium'
        input_df['absence_level'] = 'Low'
        
        key_numerical = ['G1', 'G2', 'studytime', 'absences', 'age']
        for col1 in key_numerical:
            for col2 in key_numerical:
                if col1 != col2:
                    input_df[f'poly_{col1} {col2}'] = input_df[col1] * input_df[col2]
        
        input_df['study_health_interaction'] = input_df['studytime'] * input_df['health']
        input_df['family_support_interaction'] = input_df['parent_education_avg'] * input_df['support_index']
        input_df['age_failures_interaction'] = input_df['age'] * (input_df['failures'] + 1)
        input_df['social_academic_balance'] = input_df['social_factor'] / (input_df['studytime'] + 1)
        input_df['absences_log'] = np.log1p(input_df['absences'])
        input_df['g1_g2_ratio'] = input_df['G1'] / (input_df['G2'] + 1)
        input_df['study_absence_ratio'] = input_df['studytime'] / (input_df['absences'] + 1)
        
        for col in categorical_columns:
            if col in label_encoders and col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except:
                    input_df[col] = 0
        
        missing_cols = set(feature_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        
        input_scaled = scaler.transform(input_df[feature_columns])
        
        st.markdown("""
        <style>
            .stMetric {
                background: rgba(255,255,255,0.05);
                padding: 0.8rem;        
                border-radius: 8px;
                margin: 0.3rem;           
                border: 1px solid rgba(255,255,255,0.1);
                text-align: center;
                font-size: 0.85rem;     
            }
            .stMetric > div {
                font-size: 0.9rem !important; 
            }
            .stMetric label, .stMetric span {
                font-size: 0.8rem !important;  
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.subheader("Predictions")
        
        predictions = {}
        for model_name in selected_models:
            pred = models[model_name].predict(input_scaled)[0]
            r2_value = model_scores[model_name]['R2']
            predictions[model_name] = (pred, r2_value)
        
        cols_per_row = 5
        model_items = list(predictions.items())
        
        for i in range(0, len(model_items), cols_per_row):
            row_models = model_items[i:i+cols_per_row]
            cols = st.columns(len(row_models))
            
            for j, (model_name, (pred, r2)) in enumerate(row_models):
                with cols[j]:
                    st.metric(
                        label=model_name,
                        value=f"{pred:.2f}",
                        delta=f"R²={r2:.3f}"
                    )
        
        if len(predictions) > 1:
            avg_pred = np.mean([pred for pred, r2 in predictions.values()])
            st.metric(label="Average Prediction", value=f"{avg_pred:.2f}")
        
        if len(predictions) > 1:
            st.subheader("Model Predictions Comparison")
            fig = px.bar(x=list(predictions.keys()), y=[pred for pred, r2 in predictions.values()],
                        title="Model Predictions Comparison")
            fig.update_layout(
                xaxis_title="Models", 
                yaxis_title="Predicted Grade",
                height=500,
                xaxis_tickangle=45,
                margin=dict(l=40, r=40, t=60, b=100)
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Analysis Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(data, x='G3', nbins=20, title="Grade Distribution")
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.scatter(data, x='G1', y='G3', color='sex', title="G1 vs G3 by Gender")
        st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = px.box(data, x='studytime', y='G3', title="Grades by Study Time")
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = px.scatter(data, x='absences', y='G3', title="Absences vs Final Grade")
        st.plotly_chart(fig4, use_container_width=True)
        
        fig5 = px.box(data, x='failures', y='G3', title="Grades by Past Failures")
        st.plotly_chart(fig5, use_container_width=True)
        
        fig6 = px.scatter(data, x='grade_trend', y='G3', title="Grade Trend vs Final Grade")
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        numerical_cols = ['G1', 'G2', 'G3', 'age', 'Medu', 'Fedu', 'traveltime', 
                         'studytime', 'failures', 'famrel', 'freetime', 'goout', 
                         'Dalc', 'Walc', 'health', 'absences']
        corr_matrix = data[numerical_cols].corr()
        
        fig7 = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                        title="Correlation Matrix")
        st.plotly_chart(fig7, use_container_width=True)
        
        fig8 = px.scatter(data, x='parent_education_avg', y='G3', 
                         title="Parent Education vs Grade")
        st.plotly_chart(fig8, use_container_width=True)
        
        fig9 = px.box(data, x='school', y='G3', title="Grades by School")
        st.plotly_chart(fig9, use_container_width=True)
        
        fig10 = px.scatter(data, x='motivation_score', y='G3', 
                         color='social_factor', title="Motivation vs Grade")
        st.plotly_chart(fig10, use_container_width=True)
        
        avg_grades = data.groupby('support_index')['G3'].mean().reset_index()
        fig11 = px.bar(avg_grades, x='support_index', y='G3', 
                      title="Average Grade by Support Index")
        st.plotly_chart(fig11, use_container_width=True)
        
        fig12 = px.box(data, x='health', y='G3', title="Grades by Health Status")
        st.plotly_chart(fig12, use_container_width=True)

with tab3:
    st.header("Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scores_df = pd.DataFrame(model_scores).T
        scores_df.reset_index(inplace=True)
        scores_df.rename(columns={'index': 'Model'}, inplace=True)
        
        fig13 = px.bar(scores_df, x='Model', y='R2', title="R² Score by Model")
        fig13.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig13, use_container_width=True)
        
        fig14 = px.bar(scores_df, x='Model', y='MAE', title="Mean Absolute Error by Model")
        fig14.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig14, use_container_width=True)
        
        fig15 = px.bar(scores_df, x='Model', y='MSE', title="Mean Squared Error by Model")
        fig15.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig15, use_container_width=True)
        
        st.subheader("Model Scores Table")
        st.dataframe(scores_df)
    
    with col2:
        best_model_name = max(model_scores.items(), key=lambda x: x[1]['R2'])[0]
        best_model = models[best_model_name]
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = best_model.predict(X_test_scaled)
        
        fig16 = px.scatter(x=y_test, y=y_pred, title=f"Actual vs Predicted - {best_model_name}")
        fig16.add_shape(type="line", x0=min(y_test), y0=min(y_test), 
                       x1=max(y_test), y1=max(y_test), line=dict(dash="dash"))
        fig16.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig16, use_container_width=True)
        
        residuals = y_test - y_pred
        fig17 = px.scatter(x=y_pred, y=residuals, title="Residual Plot")
        fig17.add_hline(y=0, line_dash="dash")
        fig17.update_layout(xaxis_title="Predicted", yaxis_title="Residuals")
        st.plotly_chart(fig17, use_container_width=True)
        
        if feature_importance:
            importance_data = []
            for model_name, importance in feature_importance.items():
                for feature, value in importance.items():
                    importance_data.append({'Model': model_name, 'Feature': feature, 'Importance': value})
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                top_features = importance_df.groupby('Feature')['Importance'].mean().nlargest(10)
                
                fig18 = px.bar(x=top_features.values, y=top_features.index, 
                              orientation='h', title="Top 10 Feature Importance",
                              labels={'x': 'Importance', 'y': 'Features'})
                st.plotly_chart(fig18, use_container_width=True)
        
        cross_val_scores = []
        X_train_scaled = scaler.transform(X_train)
        for name, model in list(models.items())[:5]:
            try:
                scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                cross_val_scores.append({'Model': name, 'Mean_CV_Score': scores.mean(), 'Std_CV_Score': scores.std()})
            except:
                pass
        
        if cross_val_scores:
            cv_df = pd.DataFrame(cross_val_scores)
            fig19 = px.bar(cv_df, x='Model', y='Mean_CV_Score', 
                          error_y='Std_CV_Score', title="Cross-Validation Scores")
            st.plotly_chart(fig19, use_container_width=True)

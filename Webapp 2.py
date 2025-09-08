import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Student Performance Prediction", layout="wide")

@st.cache_data
def load_model_data():
    try:
        with open('model_for_Web2.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please run code.ipynb first for model_for_Web2.pkl.")
        return None

def preprocess_input(input_data, model_data):
    df_input = pd.DataFrame([input_data])
    
    df_input['total_study_time'] = df_input['studytime'] + df_input['freetime']
    df_input['parent_education_avg'] = (df_input['Medu'] + df_input['Fedu']) / 2
    df_input['grade_trend'] = df_input['G2'] - df_input['G1']
    df_input['attendance_ratio'] = 1 - (df_input['absences'] / model_data['raw_data']['absences'].max())
    df_input['support_index'] = (df_input['schoolsup'].replace({'yes': 1, 'no': 0}) +
                                df_input['famsup'].replace({'yes': 1, 'no': 0}) +
                                df_input['paid'].replace({'yes': 1, 'no': 0}))
    
    df_input['family_quality'] = df_input['famrel'] + df_input['Pstatus'].replace({'T': 2, 'A': 0})
    df_input['social_factor'] = df_input['goout'] + df_input['Dalc'] + df_input['Walc']
    df_input['motivation_score'] = (df_input['higher'].replace({'yes': 2, 'no': 0}) +
                                   df_input['internet'].replace({'yes': 1, 'no': 0}) +
                                   df_input['romantic'].replace({'yes': -1, 'no': 1}))
    
    df_input['study_health_interaction'] = df_input['studytime'] * df_input['health']
    df_input['family_support_interaction'] = df_input['parent_education_avg'] * df_input['support_index']
    df_input['age_failures_interaction'] = df_input['age'] * (df_input['failures'] + 1)
    df_input['social_academic_balance'] = df_input['social_factor'] / (df_input['studytime'] + 1)
    
    df_input['absences_log'] = np.log1p(df_input['absences'])
    df_input['g1_g2_ratio'] = df_input['G1'] / (df_input['G2'] + 1)
    df_input['study_absence_ratio'] = df_input['studytime'] / (df_input['absences'] + 1)
    
    for col in model_data['categorical_columns']:
        if col in df_input.columns:
            le = model_data['label_encoders'][col]
            try:
                df_input[col + '_encoded'] = le.transform(df_input[col])
            except ValueError:
                df_input[col + '_encoded'] = 0
    
    return df_input[model_data['feature_columns']]

def main():
    st.title("Student Performance Prediction Dashboard")
    st.caption("A Athena Award Project, Used all light weight high accuracy model derived from my analysis if you want highest accuracy use the below link for .pkl file")
    st.caption("Uses Dataset as Source File")
    st.markdown("[Click Here For Source Code](https://github.com/lucks-13/student-performance)", unsafe_allow_html=True)
    
    model_data = load_model_data()
    if model_data is None:
        return
    
    tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Comparison"])
    
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
        
        model_list = list(model_data['models'].keys())
        
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
            
            for col in model_data['categorical_columns']:
                if col in model_data['label_encoders'] and col in input_df.columns:
                    try:
                        input_df[col] = model_data['label_encoders'][col].transform(input_df[col])
                    except:
                        input_df[col] = 0
            
            missing_cols = set(model_data['feature_columns']) - set(input_df.columns)
            for col in missing_cols:
                input_df[col] = 0
            
            input_scaled = model_data['scaler'].transform(input_df[model_data['feature_columns']])
            
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
                pred = model_data['models'][model_name].predict(input_scaled)[0]
                r2_value = model_data['model_scores'][model_name]['R2']
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
        
        df = model_data['raw_data']
        df_processed = model_data['processed_data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(df, x='G3', title='Final Grade Distribution', 
                              nbins=20, color='subject')
            st.plotly_chart(fig1)
            
            fig2 = px.scatter(df, x='G1', y='G3', color='subject',
                            title='First Grade vs Final Grade')
            st.plotly_chart(fig2)
            
            fig3 = px.box(df, x='school', y='G3', title='Grades by School')
            st.plotly_chart(fig3)
            
            fig4 = px.box(df, x='sex', y='G3', title='Grades by Gender')
            st.plotly_chart(fig4)
            
            grade_by_study = df.groupby('studytime')['G3'].mean().reset_index()
            fig5 = px.bar(grade_by_study, x='studytime', y='G3',
                         title='Average Grade by Study Time')
            st.plotly_chart(fig5)
        
        with col2:
            numerical_cols = df.select_dtypes(include=[np.number]).columns[:10]
            corr_matrix = df[numerical_cols].corr()
            fig6 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title='Correlation Matrix')
            st.plotly_chart(fig6)
            
            fig7 = px.scatter(df, x='G2', y='G3', color='failures',
                            title='Second Grade vs Final Grade (by Failures)')
            st.plotly_chart(fig7)
            
            fig8 = px.box(df, x='Medu', y='G3', title='Grades by Mother Education')
            st.plotly_chart(fig8)
            
            failure_dist = df['failures'].value_counts().sort_index()
            fig9 = px.bar(x=failure_dist.index, y=failure_dist.values,
                         title='Distribution of Past Failures')
            fig9.update_layout(xaxis_title="Number of Failures", yaxis_title="Count")
            st.plotly_chart(fig9)
            
            fig10 = px.scatter(df, x='absences', y='G3', color='subject',
                             title='Absences vs Final Grade')
            st.plotly_chart(fig10)
    
    with tab3:
        st.header("Model Performance Comparison")
        
        scores_df = pd.DataFrame(model_data['model_scores']).T
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_r2 = px.bar(scores_df, y=scores_df.index, x='R2',
                           title='R² Score Comparison', orientation='h')
            st.plotly_chart(fig_r2)
            
            fig_mse = px.bar(scores_df, y=scores_df.index, x='MSE',
                           title='MSE Comparison', orientation='h')
            st.plotly_chart(fig_mse)
        
        with col2:
            fig_mae = px.bar(scores_df, y=scores_df.index, x='MAE',
                           title='MAE Comparison', orientation='h')
            st.plotly_chart(fig_mae)
            
            st.subheader("Model Performance Table")
            st.dataframe(scores_df.round(4))
        
        st.subheader("Feature Importance Analysis")
        selected_model_fi = st.selectbox("Select model for feature importance:",
                                       ['Random Forest', 'Gradient Boosting', 'XGBoost'])
        
        if selected_model_fi in model_data['models']:
            model = model_data['models'][selected_model_fi]
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': model_data['feature_columns'],
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(15)
                
                fig_importance = px.bar(importance_df, y='feature', x='importance',
                                      title=f'{selected_model_fi} - Top 15 Feature Importance',
                                      orientation='h')
                st.plotly_chart(fig_importance)
        
        st.subheader("Model Predictions vs Actual")
        selected_comparison_model = st.selectbox("Select model for comparison:",
                                                list(model_data['models'].keys()))
        
        model = model_data['models'][selected_comparison_model]
        X_test_scaled = model_data['scaler'].transform(model_data['X_test'])
        y_pred = model.predict(X_test_scaled)
        y_test = model_data['y_test']
        
        fig_pred = px.scatter(x=y_test, y=y_pred, title=f'{selected_comparison_model} - Predictions vs Actual')
        fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                    y=[y_test.min(), y_test.max()], 
                                    mode='lines', name='Perfect Prediction'))
        fig_pred.update_layout(xaxis_title="Actual Values", yaxis_title="Predicted Values")
        st.plotly_chart(fig_pred)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap
import os
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURACIÓN GENERAL
# ==============================================================================
st.set_page_config(
    page_title="Predicción de Ingresos AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. BACKEND Y LÓGICA ML 
# ==============================================================================

@st.cache_resource
def load_artifacts():
    """Carga modelo y artefactos con manejo de errores."""
    artifacts = {}
    try:
        # Intentamos cargar los archivos generados localmente
        artifacts['model'] = tf.keras.models.load_model('modelo_ingresos.keras')
        artifacts['preprocessor'] = joblib.load('preprocessor.joblib')
        artifacts['background'] = joblib.load('shap_background.joblib')
        artifacts['cols'] = joblib.load('columnas_input.joblib')
        artifacts['feat_names'] = artifacts['preprocessor'].get_feature_names_out()
        return artifacts
    except FileNotFoundError as e:
        st.error(f"Error Crítico: Faltan archivos. Ejecuta 'entrenar_local.py' primero. {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error desconocido: {e}")
        st.stop()

arts = load_artifacts()

def make_prediction(input_df):
    """Pipeline de predicción corregido."""
    try:
        # 1. Preprocesar
        X_processed = arts['preprocessor'].transform(input_df).toarray()
        
        # 2. Predecir
        prob = arts['model'].predict(X_processed, verbose=0)[0][0]
        prediction = 1 if prob > 0.5 else 0
        
        # 3. Explicabilidad (SHAP)
        explainer = shap.DeepExplainer(arts['model'], arts['background'])
        shap_values = explainer.shap_values(X_processed)
        
        # --- CORRECCIÓN DE DIMENSIONES SHAP ---
        # 1. Sacar de la lista si es necesario
        if isinstance(shap_values, list): 
            shap_values = shap_values[0]
            
        # 2. Aplanar dimensiones extra (de (1, 87) a (87,))
        if shap_values.ndim > 1:
            shap_values = shap_values.flatten()

        # --- CORRECCIÓN DE BASE_VALUE (El error que tenías) ---
        base_value = explainer.expected_value
        # Desempaquetar recursivamente hasta encontrar el número
        while isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
        
        # Convertir explícitamente a float para evitar "setting array element..."
        base_value = float(base_value)

        return prediction, prob, shap_values, base_value, X_processed[0]
        
    except Exception as e:
        st.error(f"Error técnico en predicción: {e}")
        return None, None, None, None, None

def log_prediction(data_dict, prediction, prob):
    """Guarda logs."""
    try:
        log_data = data_dict.copy()
        log_data['Prediction'] = '>50K' if prediction == 1 else '<=50K'
        log_data['Probability'] = round(prob, 4)
        df_log = pd.DataFrame([log_data])
        
        header = not os.path.isfile('prediction_logs.csv')
        df_log.to_csv('prediction_logs.csv', mode='a', header=header, index=False)
    except Exception:
        pass

# ==============================================================================
# 2. FRONTEND
# ==============================================================================

st.title("Sistema de Predicción de Ingresos")
st.markdown("Estime la probabilidad de ingresos >$50K anuales basado en datos del censo.")

# --- Formulario ---
st.sidebar.header("Perfil")

def user_input_features():
    age = st.sidebar.slider('Edad', 17, 90, 30, help="Años")
    workclass = st.sidebar.selectbox('Clase de Trabajo', 
        ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    education_num = st.sidebar.slider('Años de Educación', 1, 16, 10, help="13=Bachelors")
    marital_status = st.sidebar.selectbox('Estado Civil',
        ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = st.sidebar.selectbox('Ocupación',
        ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    relationship = st.sidebar.selectbox('Rol Familiar',
        ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = st.sidebar.selectbox('Raza', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    sex = st.sidebar.selectbox('Sexo', ['Female', 'Male'])
    capital_gain = st.sidebar.number_input('Ganancia Capital', 0, 99999, 0)
    capital_loss = st.sidebar.number_input('Pérdida Capital', 0, 4356, 0)
    
    if capital_gain > 0 and capital_loss > 0:
        st.sidebar.warning("Aviso: Ganancia y pérdida simultánea es inusual.")

    hours_per_week = st.sidebar.slider('Horas/Semana', 1, 99, 40)
    native_country = st.sidebar.selectbox('País', ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'China', 'South', 'Jamaica', 'Italy', 'Dominican-Republic', 'Japan', 'Guatemala', 'Poland', 'Vietnam', 'Columbia', 'Haiti', 'Portugal', 'Taiwan', 'Iran', 'Nicaragua', 'Peru', 'Ecuador', 'France', 'Greece', 'Ireland', 'Thailand', 'Hong', 'Cambodia', 'Trinadad&Tobago', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands'])

    data = {
        'age': age, 'workclass': workclass, 'education.num': education_num,
        'marital.status': marital_status, 'occupation': occupation,
        'relationship': relationship, 'race': race, 'sex': sex,
        'capital.gain': capital_gain, 'capital.loss': capital_loss,
        'hours.per.week': hours_per_week, 'native.country': native_country
    }
    return pd.DataFrame(data, index=[0]), data

input_df, input_dict = user_input_features()

# --- Resultados ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Predicción")
    if st.button('Calcular', use_container_width=True):
        with st.spinner('Analizando...'):
            prediction, prob, shap_values, base_value, x_values = make_prediction(input_df)
            
            if prediction is not None:
                log_prediction(input_dict, prediction, prob)
                
                m1, m2 = st.columns(2)
                m1.metric("Probabilidad", f"{prob:.1%}")
                # Usamos un if-else normal para evitar que Streamlit imprima el objeto técnico
                if prediction == 1:
                    m2.success("## >50K")
                else:
                    m2.warning("## <=50K")
                
                st.divider()
                st.subheader("Explicación Visual")
                st.caption("Rojo = Sube probabilidad | Azul = Baja probabilidad")
                
                try:
                    # Aquí pasamos x_values para que el gráfico muestre "Edad=30", etc.
                    st_shap_plot = shap.force_plot(
                        base_value, 
                        shap_values, 
                        x_values, # Datos reales para mostrar en el gráfico
                        feature_names=arts['feat_names'],
                        matplotlib=True, 
                        show=False
                    )
                    st.pyplot(st_shap_plot, bbox_inches='tight')
                except Exception as e:
                    st.error(f"Error gráfico: {e}")

with col2:
    st.subheader("Historial")
    if os.path.exists('prediction_logs.csv'):
        df_logs = pd.read_csv('prediction_logs.csv')
        st.write(f"Reg: {len(df_logs)}")
        if not df_logs.empty:
            st.bar_chart(df_logs['Prediction'].value_counts())
            st.dataframe(df_logs.tail(3)[['age', 'Prediction']], hide_index=True)
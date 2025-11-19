import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap
import os
import matplotlib.pyplot as plt

# ==============================================================================
# 1. BACKEND Y LÓGICA ML
# ==============================================================================

# Configuración de la página
st.set_page_config(page_title="Predicción de Ingresos AI", layout="wide")

@st.cache_resource
def load_artifacts():
    """Carga el modelo, el preprocesador y los datos de fondo para SHAP"""
    try:
        model = tf.keras.models.load_model('modelo_ingresos.keras')
        preprocessor = joblib.load('preprocessor.joblib')
        background = joblib.load('shap_background.joblib')
        cols = joblib.load('columnas_input.joblib')
        # Necesitamos los nombres de las features transformadas para SHAP
        feature_names_out = preprocessor.get_feature_names_out()
        return model, preprocessor, background, cols, feature_names_out
    except Exception as e:
        st.error(f"Error cargando artefactos: {e}")
        return None, None, None, None, None

# Cargar todo
model, preprocessor, background, input_cols, feature_names_out = load_artifacts()

def make_prediction(input_df):
    """Función que procesa los datos y devuelve la predicción y SHAP values"""
    # 1. Preprocesar
    X_processed = preprocessor.transform(input_df).toarray()
    
    # 2. Predecir
    prob = model.predict(X_processed, verbose=0)[0][0]
    prediction = 1 if prob > 0.5 else 0
    
    # 3. Explicabilidad (SHAP)
    # Usamos DeepExplainer como en el notebook
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_processed)
    
    # Ajuste de dimensiones para SHAP (aplanar si es necesario)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim > 2:
        shap_values = shap_values.squeeze()
        
    # Asegurar que shap_values sea 1D para una sola predicción
    if shap_values.ndim == 2 and shap_values.shape[0] == 1:
        shap_values = shap_values.flatten()

    base_value = explainer.expected_value
    if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
         base_value = base_value[0]

    return prediction, prob, shap_values, base_value, X_processed

def log_prediction(data_dict, prediction, prob):
    """Sistema de Monitoreo simple: Guarda las predicciones en un CSV"""
    log_data = data_dict.copy()
    log_data['Prediction'] = '>50K' if prediction == 1 else '<=50K'
    log_data['Probability'] = round(prob, 4)
    
    df_log = pd.DataFrame([log_data])
    
    if not os.path.isfile('prediction_logs.csv'):
        df_log.to_csv('prediction_logs.csv', index=False)
    else:
        df_log.to_csv('prediction_logs.csv', mode='a', header=False, index=False)

# ==============================================================================
# 2. FRONTEND (STREAMLIT)
# ==============================================================================

st.title("💰 Predicción de Ingresos (Adult Census)")
st.markdown("""
Este sistema utiliza una **Red Neuronal Profunda** para estimar si una persona gana más de $50K anuales.
""")

# --- A. Formulario de Entrada (Sidebar) ---
st.sidebar.header("📝 Perfil del Usuario")

def user_input_features():
    # Definimos los inputs basados en las columnas originales (orden importante)
    # Nota: Ajusta las opciones de las listas según tu dataset real si varían
    
    age = st.sidebar.slider('Edad', 17, 90, 30)
    
    workclass = st.sidebar.selectbox('Clase de Trabajo', 
        ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    
    # education.num suele ser más fácil de manejar que education texto
    education_num = st.sidebar.slider('Años de Educación', 1, 16, 10)
    
    marital_status = st.sidebar.selectbox('Estado Civil',
        ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    
    occupation = st.sidebar.selectbox('Ocupación',
        ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    
    relationship = st.sidebar.selectbox('Relación',
        ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    
    race = st.sidebar.selectbox('Raza',
        ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    
    sex = st.sidebar.selectbox('Sexo', ['Female', 'Male'])
    
    capital_gain = st.sidebar.number_input('Ganancia de Capital', 0, 99999, 0)
    capital_loss = st.sidebar.number_input('Pérdida de Capital', 0, 4356, 0)
    hours_per_week = st.sidebar.slider('Horas por Semana', 1, 99, 40)
    
    native_country = st.sidebar.selectbox('País de Origen', ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'China', 'South', 'Jamaica', 'Italy', 'Dominican-Republic', 'Japan', 'Guatemala', 'Poland', 'Vietnam', 'Columbia', 'Haiti', 'Portugal', 'Taiwan', 'Iran', 'Nicaragua', 'Peru', 'Ecuador', 'France', 'Greece', 'Ireland', 'Thailand', 'Hong', 'Cambodia', 'Trinadad&Tobago', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands'])

    # Crear diccionario
    data = {
        'age': age,
        'workclass': workclass,
        # 'fnlwgt': 0, # Si la eliminaste en el notebook, no la incluyas. Si está, pon un promedio.
        # 'education': ' Bachelors', # No la usamos si usamos education-num
        'education.num': education_num,
        'marital.status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital.gain': capital_gain,
        'capital.loss': capital_loss,
        'hours.per.week': hours_per_week,
        'native.country': native_country
    }
    
    # Asegurarse de que el orden coincida con el entrenamiento
    # Filtramos solo las columnas que realmente espera el modelo (input_cols)
    features = pd.DataFrame(data, index=[0])
    return features, data

input_df, input_dict = user_input_features()

# --- B. Panel Principal ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Resultado de la Predicción")
    
    if st.button('🚀 Calcular Predicción'):
        with st.spinner('Analizando datos...'):
            # Llamada al Backend
            prediction, prob, shap_values, base_value, X_processed = make_prediction(input_df)
            
            # Guardar en Log (Monitoreo)
            log_prediction(input_dict, prediction, prob)
            
            # Mostrar Resultado
            if prediction == 1:
                st.success(f"💰 **Ingresos Altos (>50K)** detectados.")
            else:
                st.warning(f"📉 **Ingresos Bajos (<=50K)** detectados.")
            
            st.metric(label="Probabilidad de >50K", value=f"{prob:.2%}")
            
            # --- Mecanismo de Explicación (SHAP) ---
            st.subheader("🔍 Explicación del Modelo (SHAP)")
            st.write("Este gráfico muestra qué factores empujaron la decisión hacia arriba (rojo) o hacia abajo (azul).")
            
            # Usamos matplotlib para dibujar el force_plot estático en Streamlit
            # (Es más estable que la versión JS en algunos entornos)
            try:
                st_shap_plot = shap.force_plot(
                    base_value,
                    shap_values,
                    feature_names=feature_names_out,
                    matplotlib=True,
                    show=False
                )
                st.pyplot(st_shap_plot, bbox_inches='tight')
            except Exception as e:
                st.error(f"No se pudo generar el gráfico SHAP: {e}")
            
            # Mostrar las 3 características más importantes en texto
            st.write("**Factores más influyentes para este caso:**")
            # Lógica simple para mostrar los top features
            indices = np.argsort(np.abs(shap_values))[::-1] # Indices de mayor impacto
            for i in range(3):
                idx = indices[i]
                feature_name = feature_names_out[idx]
                impact = shap_values[idx]
                direction = "Positivo (Sube ingreso)" if impact > 0 else "Negativo (Baja ingreso)"
                st.info(f"{i+1}. **{feature_name}**: Impacto {direction}")


# --- C. Panel de Monitoreo (Requisito del docente) ---
with col2:
    st.subheader("📊 Monitoreo")
    if os.path.exists('prediction_logs.csv'):
        df_logs = pd.read_csv('prediction_logs.csv')
        
        st.write(f"Total Predicciones: **{len(df_logs)}**")
        
        if len(df_logs) > 0:
            # Gráfico simple de distribución de predicciones
            st.write("Distribución:")
            st.bar_chart(df_logs['Prediction'].value_counts())
            
            st.write("Últimos 5 registros:")
            st.dataframe(df_logs.tail(5)[['age', 'occupation', 'Prediction']], hide_index=True)
    else:
        st.write("Aún no hay registros.")
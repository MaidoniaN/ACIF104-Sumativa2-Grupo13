import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import shap

print("--- 1. Cargando y Limpiando Datos ---")
# Cargar datos
df = pd.read_csv('adult.csv')

# Limpieza (Misma lógica del notebook)
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

df.replace('?', np.nan, inplace=True)
df.dropna(subset=['income'], inplace=True)

# Eliminar columnas redundantes
df.drop(columns=['education', 'fnlwgt'], inplace=True, errors='ignore')

# Separar X e y
X = df.drop(columns=['income'])
y = df['income'].map({'<=50K': 0, '>50K': 1})

# Tipos de columnas
numerical_cols = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_cols = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("--- 2. Creando Preprocesador ---")
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Ajustar preprocesador
X_train_prep = preprocessor.fit_transform(X_train).toarray()
input_dim = X_train_prep.shape[1]

# Calcular pesos de clase
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_dict = {0: weights[0], 1: weights[1]}

print(f"--- 3. Entrenando Modelo (MLP Dropout) ---")
# Arquitectura Ganadora
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar (Rápido, 15 épocas es suficiente para la demo)
model.fit(X_train_prep, y_train, epochs=15, batch_size=64, class_weight=weights_dict, verbose=1)

print("--- 4. Guardando Artefactos Compatibles ---")

# Guardar Modelo
model.save('modelo_ingresos.keras')

# Guardar Preprocesador (Aquí se soluciona tu error)
joblib.dump(preprocessor, 'preprocessor.joblib')

# Guardar nombres de columnas para el frontend
joblib.dump(X.columns.tolist(), 'columnas_input.joblib')

# Guardar fondo para SHAP (tomamos una muestra)
X_train_summary = shap.sample(X_train_prep, 50)
joblib.dump(X_train_summary, 'shap_background.joblib')

print("¡LISTO! Archivos generados correctamente para este entorno.")
## 🔧 Guía de Despliegue y Ejecución

### 1. Prerrequisitos del Sistema
Este proyecto fue desarrollado en **Python 3.10+**. Asegúrate de tener instalado Python y `pip` en tu sistema.

### 2. Instalación
Sigue estos pasos para configurar el entorno de ejecución:

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/MaidoniaN/ACIF104-Sumativa1-Grupo1.git](https://github.com/MaidoniaN/ACIF104-Sumativa1-Grupo1.git)
    cd ACIF104-Sumativa1-Grupo1
    ```

2.  **Configurar entorno virtual (Recomendado):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # En Windows usar: venv\Scripts\activate
    ```

3.  **Instalar dependencias:**
    Todas las librerías necesarias están listadas en `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### 3. Ejecución de la Aplicación (Backend + Frontend)
Para lanzar el prototipo funcional, navega a la carpeta de la API y ejecuta Streamlit:

1.  Ir al directorio de la aplicación:
    ```bash
    cd API
    ```

2.  **(Solo primera vez) Generar artefactos locales:**
    Para asegurar compatibilidad, ejecuta el script de entrenamiento ligero:
    ```bash
    python3 entrenar_local.py
    ```
    *Espera el mensaje: "¡LISTO! Archivos generados correctamente."*

3.  **Iniciar el servidor:**
    ```bash
    streamlit run app.py
    ```

4.  **Acceso:**
    La aplicación se abrirá automáticamente en tu navegador en: `http://localhost:8501`

### 4. Estructura del Proyecto
El repositorio está modularizado para facilitar el mantenimiento:
* `ACIF104_S6_Grupo13.ipynb`: Notebook principal con el análisis (EDA), modelado y evaluación.
* `API/`: Carpeta contenedora del despliegue.
    * `app.py`: Código fuente del sistema (Frontend Streamlit + Backend TensorFlow).
    * `entrenar_local.py`: Script auxiliar para regenerar modelos compatibles.
    * `prediction_logs.csv`: Archivo de registro para el monitoreo de predicciones.
* `requirements.txt`: Lista de dependencias para reproducibilidad.
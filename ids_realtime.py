import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tcn import TCN
import random
import pyarrow.feather as feather


# Configuración de colores para la consola
class Colors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def cargar_artefactos():
    print("Cargando sistema IDS...")
    # Cargar el modelo TCN (asegúrate de que el archivo .keras esté en la misma carpeta)
    try:
        model = tf.keras.models.load_model('mejor_modelo_tcn.keras', custom_objects={'TCN': TCN})
        scaler = joblib.load('scaler_entrenado.joblib')
        le = joblib.load('label_encoder_entrenado.joblib')
        print(f"{Colors.OKGREEN}Sistema cargado correctamente.{Colors.ENDC}")
        return model, scaler, le
    except Exception as e:
        print(f"{Colors.FAIL}Error cargando artefactos: {e}{Colors.ENDC}")
        exit()


def simular_trafico_real(df_holdout, model, scaler, le):
    print(f"\n{Colors.BOLD}Iniciando Monitor de Red (Simulado)...{Colors.ENDC}")
    print("Presiona CTRL+C para detener.\n")

    # Separar features y etiquetas reales (para comprobar si acertamos)
    y_true = df_holdout['label_encoded'].values
    X_raw = df_holdout.drop(columns=['label_encoded'])

    # Obtener nombres de columnas esperadas por el scaler
    # Esto es crucial para mantener el orden correcto
    columnas_modelo = scaler.feature_names_in_
    X_raw = X_raw[columnas_modelo]

    n_muestras = len(X_raw)

    try:
        while True:
            # 1. Simular la llegada de un nuevo flujo (elegir uno al azar del holdout)
            idx = random.randint(0, n_muestras - 1)

            # Extraer la fila como DataFrame (para mantener nombres de columnas)
            flujo_entrante = X_raw.iloc[[idx]]
            etiqueta_real_num = y_true[idx]
            etiqueta_real_txt = le.inverse_transform([etiqueta_real_num])[0]

            # 2. Preprocesamiento (El mismo que en el entrenamiento)
            # Escalar
            flujo_scaled = scaler.transform(flujo_entrante)

            # Reshape para TCN (1 muestra, 1 timestep, N features)
            flujo_3d = flujo_scaled.reshape((1, 1, flujo_scaled.shape[1]))

            # 3. Inferencia
            inicio_inf = time.time()
            pred_probs = model.predict(flujo_3d, verbose=0)
            latencia = (time.time() - inicio_inf) * 1000  # ms

            pred_idx = np.argmax(pred_probs)
            confianza = np.max(pred_probs)
            pred_txt = le.inverse_transform([pred_idx])[0]

            # 4. Mostrar resultados / Alertas
            es_ataque = pred_txt != "BenignTraffic"

            if es_ataque:
                status_color = Colors.FAIL
                icono = "🚨 ATAQUE DETECTADO"
            else:
                status_color = Colors.OKGREEN
                icono = "✅ Tráfico Normal"

            # Imprimir log tipo consola de seguridad
            print(f"[{time.strftime('%H:%M:%S')}] FlowID: {idx} | "
                  f"{icono} : {status_color}{pred_txt}{Colors.ENDC} "
                  f"(Confianza: {confianza:.2%} | Latencia: {latencia:.1f}ms)")

            # Verificación (Solo para fines académicos/demo)
            if pred_txt != etiqueta_real_txt:
                print(f"   ⚠️  Predicción incorrecta. Real: {etiqueta_real_txt}")

            # Simular tiempo entre paquetes (para que sea legible)
            time.sleep(random.uniform(0.5, 2.0))

    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Monitor detenido por el usuario.{Colors.ENDC}")


if __name__ == "__main__":
    # Cargar datos reservados
    try:
        df_holdout = feather.read_feather('holdout_data.feather')
    except:
        print("No se encontró 'holdout_data.feather'. Ejecuta el notebook de entrenamiento primero.")
        exit()

    model, scaler, le = cargar_artefactos()
    simular_trafico_real(df_holdout, model, scaler, le)
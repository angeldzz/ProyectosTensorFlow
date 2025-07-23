import cv2  # Biblioteca para procesamiento de imágenes y video.
import mediapipe as mp  # Biblioteca para detección de manos.
import tensorflow as tf  # Biblioteca para trabajar con modelos de aprendizaje automático.
import numpy as np  # Biblioteca para manipulación de datos numéricos.

# Carga el modelo de TensorFlow previamente entrenado para clasificar gestos.
modelo = tf.keras.models.load_model("gestos_modelo.h5")

# Lista de nombres de gestos (debe coincidir con las clases del modelo entrenado).
gestos = ["hola","bien","mal","piedra","spock","lagarto"]

# Inicializa los módulos de MediaPipe para detección de manos.
mp_hands = mp.solutions.hands  # Carga el módulo de detección de manos.
hands = mp_hands.Hands()  # Crea un objeto para procesar imágenes y detectar manos.
mp_draw = mp.solutions.drawing_utils  # Utilidad para dibujar las conexiones de las manos detectadas.

# Inicia la captura de video desde la cámara (índice 0 para la cámara predeterminada).
cap = cv2.VideoCapture(0)

# Bucle principal para procesar cada cuadro del video.
while True:
    success, img = cap.read()  # Lee un cuadro de la cámara.
    if not success:  # Si no se puede leer el cuadro, se detiene el bucle.
        break

    # Convierte la imagen de BGR (formato de OpenCV) a RGB (formato requerido por MediaPipe).
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)  # Procesa la imagen para detectar manos.

    # Si se detectan manos en la imagen:
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # Itera sobre cada mano detectada.
            # Dibuja los puntos de referencia (landmarks) y las conexiones en la imagen original.
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Extrae las coordenadas de los puntos de referencia (landmarks).
            landmarks = []
            for lm in handLms.landmark:
                landmarks.append([lm.x, lm.y, lm.z])  # Normaliza las coordenadas.

            # Convierte las coordenadas a un array de NumPy y aplana la lista.
            landmarks_array = np.array(landmarks).flatten()

            # Realiza una predicción con el modelo.
            prediccion = modelo.predict(np.expand_dims(landmarks_array, axis=0))
            gesto_id = np.argmax(prediccion)  # Obtiene el índice del gesto con mayor probabilidad.
            gesto_detectado = gestos[gesto_id]  # Obtiene el nombre del gesto.

            # Muestra el gesto detectado en la imagen.
            cv2.putText(img, f"Gesto: {gesto_detectado}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Muestra la imagen procesada en una ventana llamada "Detector de Gestos".
    cv2.imshow("Detector de Gestos", img)
    # Espera a que se presione la tecla 'q' para salir del bucle.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas abiertas por OpenCV.
cap.release()
cv2.destroyAllWindows()

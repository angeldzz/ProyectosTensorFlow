import cv2
import mediapipe as mp
import os
import csv

GESTO = "tijera"  # Cambia este valor por el nombre del gesto actual que quieres grabar

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Crear carpeta si no existe
os.makedirs("ProyectosTensorFlow/DetectorManos/datos", exist_ok=True)

# Archivo CSV para guardar los datos
archivo_csv = f"ProyectosTensorFlow/DetectorManos/datos/{GESTO}.csv"
f = open(archivo_csv, mode="a", newline='')
writer = csv.writer(f)

print(f"[INFO] Recolectando datos para: {GESTO}")
print("[INFO] Pulsa 's' para guardar un ejemplo. Pulsa 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo capturar el cuadro de la cámara.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            puntos = []
            for lm in hand_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])

            # Guardar si se pulsa 's'
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print(f"Puntos capturados: {puntos}")  # Verifica si hay datos
                writer.writerow(puntos)
                print(f"[GUARDADO] Ejemplo de '{GESTO}' añadido.")

    cv2.imshow("Recolectar Gesto", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

f.close()
cap.release()
cv2.destroyAllWindows()

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Ruta de la carpeta con los datos
carpeta_datos = "ProyectosTensorFlow/DetectorManos/datos"

# Lista de gestos (debe coincidir con los nombres de los archivos CSV)
gestos = ["papel","tijera","piedra", "lagarto", "bien", "mal"]

# Cargar los datos desde los archivos CSV
def cargar_datos(carpeta, gestos):
    X = []
    y = []
    for idx, gesto in enumerate(gestos):
        archivo = os.path.join(carpeta, f"{gesto}.csv")
        if os.path.exists(archivo):
            with open(archivo, "r") as f:
                for linea in f:
                    puntos = list(map(float, linea.strip().split(",")))
                    X.append(puntos)
                    y.append(idx)  # Etiqueta numérica para el gesto
    return np.array(X), np.array(y)

# Cargar los datos
X, y = cargar_datos(carpeta_datos, gestos)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Asegurarse de que los datos tengan la forma correcta
print(f"Datos de entrenamiento: {X_train.shape}, Etiquetas: {y_train.shape}")
print(f"Datos de prueba: {X_test.shape}, Etiquetas: {y_test.shape}")

# Definir el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu"),  # Aumenta el número de neuronas
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(gestos), activation="softmax")
])

# Compilar el modelo con un learning rate ajustado
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Resumen del modelo
modelo.summary()

# Entrenar el modelo
historial = modelo.fit(X_train, y_train, epochs=2000, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo entrenado
modelo.save("ProyectosTensorFlow/DetectorManos/gestos_modelo_v2.keras")
print("Modelo guardado como 'gestos_modelo_v2.keras'")

# Evaluar el modelo
loss, accuracy = modelo.evaluate(X_test, y_test)
print(f"Pérdida: {loss}, Precisión: {accuracy}")
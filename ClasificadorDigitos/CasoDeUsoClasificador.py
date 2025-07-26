import tensorflow as tf
import numpy as np
from keras.models import load_model
from PIL import Image
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

# Cargar el modelo guardado
model = load_model('ProyectosTensorFlow/ClasificadorDigitos/clasificador_mnist.keras')

# Función para preprocesar la imagen
def preprocess_image(image_path):
    # Abrir la imagen
    img = Image.open(image_path).convert('L')  # Convertir a escala de grises
    img = img.resize((28, 28))  # Redimensionar a 28x28 píxeles
    img = np.array(img)  # Convertir a un array de NumPy
    img = 255 - img  # Invertir colores (MNIST tiene fondo negro y dígitos blancos)
    img = img.astype('float32') / 255.0  # Normalizar a valores entre 0 y 1
    img = img.reshape(1, 28, 28, 1)  # Añadir dimensiones para el modelo
    return img

# Bucle para seleccionar y comprobar múltiples imágenes
while True:
    # Abrir un cuadro de diálogo para seleccionar la imagen
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal
    image_path = filedialog.askopenfilename(
        title="Selecciona una imagen (o cancela para salir)",
        filetypes=[("Imagenes PNG", "*.png"), ("Todos los archivos", "*.*")]
    )
    root.destroy()

    # Si no se selecciona ninguna imagen, salir del bucle
    if not image_path:
        print("No se seleccionó ninguna imagen. Saliendo...")
        break

    # Preprocesar la imagen
    input_image = preprocess_image(image_path)

    # Realizar la predicción
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions)

    # Mostrar el resultado
    print(f"El modelo predice que el dígito es: {predicted_class}")

    # Mostrar la imagen
    plt.imshow(input_image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicción: {predicted_class}")
    plt.axis('off')
    plt.show()
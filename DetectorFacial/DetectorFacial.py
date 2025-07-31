import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

# Modelo preentrenado para detección de emociones basado en dataset FER2013

class DetectorExpresiones:
    def __init__(self, modelo_path="ProyectosTensorflow/DetectorFacial/face_model.h5"):
        """
        Inicializa el detector de expresiones faciales
        
        Args:
            modelo_path (str): Ruta al modelo preentrenado basado en FER2013
        """
        # Cargar el modelo preentrenado
        self.modelo = load_model(modelo_path)
        
        # Definir las etiquetas de emociones según FER2013 (7 categorías)
        self.etiquetas_emociones = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        # Cargar el clasificador de caras de OpenCV
        self.detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def preprocesar_imagen(self, imagen_cara):
        """
        Preprocesa la imagen de la cara para el modelo FER2013 (48x48 grayscale)
        
        Args:
            imagen_cara: Imagen de la cara detectada
            
        Returns:
            Imagen preprocesada para el modelo (48x48 normalizada)
        """
        # Redimensionar a 48x48 (tamaño estándar del dataset FER2013)
        imagen_cara = cv2.resize(imagen_cara, (48, 48))
        # Normalizar valores de píxeles a rango [0, 1]
        imagen_cara = imagen_cara.astype("float") / 255.0
        # Convertir a array y añadir dimensión de batch
        imagen_cara = img_to_array(imagen_cara)
        imagen_cara = np.expand_dims(imagen_cara, axis=0)
        
        return imagen_cara
    
    def detectar_emociones_imagen(self, ruta_imagen):
        """
        Detecta emociones en una imagen estática
        
        Args:
            ruta_imagen (str): Ruta a la imagen
            
        Returns:
            Imagen con las emociones detectadas
        """
        # Cargar la imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
            return None
            
        imagen_original = imagen.copy()
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras con parámetros optimizados
        caras = self.detector_caras.detectMultiScale(
            imagen_gris, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"Caras detectadas: {len(caras)}")
        
        for (x, y, w, h) in caras:
            # Extraer la región de la cara
            cara = imagen_gris[y:y+h, x:x+w]
            cara_procesada = self.preprocesar_imagen(cara)
            
            # Predecir la emoción
            prediccion = self.modelo.predict(cara_procesada, verbose=0)[0]
            indice_emocion = np.argmax(prediccion)
            etiqueta = self.etiquetas_emociones[indice_emocion]
            confianza = prediccion[indice_emocion]
            
            # Configurar colores según la emoción detectada
            color = self.obtener_color_emocion(etiqueta)
            
            # Dibujar el rectángulo y la etiqueta
            cv2.rectangle(imagen, (x, y), (x+w, y+h), color, 2)
            cv2.putText(imagen, f"{etiqueta}: {confianza:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return imagen
    
    def obtener_color_emocion(self, emocion):
        """
        Retorna un color específico para cada emoción
        """
        colores = {
            "Angry": (0, 0, 255),
            "Disgust": (0, 100, 0),
            "Fear": (128, 0, 128),
            "Happy": (0, 255, 255),
            "Sad": (255, 0, 0),
            "Surprise": (255, 165, 0),
            "Neutral": (0, 255, 0)
        }

        return colores.get(emocion, (0, 255, 0))
    
    def detectar_emociones_tiempo_real(self):
        """
        Detecta emociones en tiempo real usando la webcam
        """
        # Inicializar la webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara")
            return
        
        print("=== DETECTOR DE EXPRESIONES FACIALES EN TIEMPO REAL ===")
        print("Presiona 'q' para salir...")
        print("Emociones detectables: Enfadado, Disgustado, Miedo, Feliz, Triste, Sorprendido, Neutral")
        
        # Contador de frames para optimizar rendimiento
        frame_count = 0
        
        while True:
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar cada 3 frames para mejorar rendimiento
            frame_count += 1
            if frame_count % 3 != 0:
                cv2.imshow('Detector de Expresiones Faciales - FER2013', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Convertir a escala de grises
            frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar caras
            caras = self.detector_caras.detectMultiScale(
                frame_gris, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in caras:
                # Extraer la región de la cara
                cara = frame_gris[y:y+h, x:x+w]
                cara_procesada = self.preprocesar_imagen(cara)
                
                # Predecir la emoción
                prediccion = self.modelo.predict(cara_procesada, verbose=0)[0]
                indice_emocion = np.argmax(prediccion)
                etiqueta = self.etiquetas_emociones[indice_emocion]
                confianza = prediccion[indice_emocion]
                
                # Obtener color para la emoción
                color = self.obtener_color_emocion(etiqueta)
                
                # Dibujar el rectángulo y la etiqueta
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{etiqueta}: {confianza:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Mostrar todas las probabilidades (opcional)
                y_offset = y + h + 20
                for i, prob in enumerate(prediccion):
                    if prob > 0.1:  # Solo mostrar probabilidades significativas
                        texto_prob = f"{self.etiquetas_emociones[i]}: {prob:.2f}"
                        cv2.putText(frame, texto_prob, (x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        y_offset += 15
            
            # Mostrar el frame
            cv2.imshow('Detector de Expresiones Faciales - FER2013', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Limpiar recursos
        cap.release()
        cv2.destroyAllWindows()
    
    def detectar_emociones_video(self, ruta_video, guardar_resultado=False):
        """
        Detecta emociones en un archivo de video
        
        Args:
            ruta_video (str): Ruta al archivo de video
            guardar_resultado (bool): Si guardar el video procesado
        """
        cap = cv2.VideoCapture(ruta_video)
        
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {ruta_video}")
            return
        
        # Configurar el video de salida si se desea guardar
        if guardar_resultado:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter('video_procesado.avi', fourcc, fps, (width, height))
        
        print("Procesando video... Presiona 'q' para salir")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            caras = self.detector_caras.detectMultiScale(frame_gris, scaleFactor=1.1, 
                                                        minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in caras:
                cara = frame_gris[y:y+h, x:x+w]
                cara = self.preprocesar_imagen(cara)
                
                prediccion = self.modelo.predict(cara)[0]
                etiqueta = self.etiquetas_emociones[np.argmax(prediccion)]
                confianza = np.max(prediccion)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{etiqueta}: {confianza:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if guardar_resultado:
                out.write(frame)
            
            cv2.imshow('Detector de Expresiones - Video', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if guardar_resultado:
            out.release()
        cv2.destroyAllWindows()


def main():
    """Función principal con ejemplos de uso"""
    
    # Verificar que existe el modelo
    if not os.path.exists("ProyectosTensorflow/DetectorFacial/face_model.h5"):
        print("Error: No se encontró el archivo ProyectosTensorflow/DetectorFacial/face_model.h5")
        print("Asegúrate de que el modelo esté en la misma carpeta que este script")
        return
    
    # Crear el detector
    detector = DetectorExpresiones("ProyectosTensorflow/DetectorFacial/face_model.h5")
    
    print("=== DETECTOR DE EXPRESIONES FACIALES ===")
    print("1. Detección en tiempo real (webcam)")
    print("2. Detección en imagen")
    print("3. Detección en video")
    print("4. Salir")
    
    while True:
        opcion = input("\nSelecciona una opción (1-4): ")
        
        if opcion == "1":
            detector.detectar_emociones_tiempo_real()
            
        elif opcion == "2":
            ruta_imagen = input("Ingresa la ruta de la imagen: ")
            if os.path.exists(ruta_imagen):
                resultado = detector.detectar_emociones_imagen(ruta_imagen)
                cv2.imshow('Detección de Emociones', resultado)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("La imagen no existe")
                
        elif opcion == "3":
            ruta_video = input("Ingresa la ruta del video: ")
            if os.path.exists(ruta_video):
                guardar = input("¿Guardar video procesado? (s/n): ").lower() == 's'
                detector.detectar_emociones_video(ruta_video, guardar)
            else:
                print("El video no existe")
                
        elif opcion == "4":
            print("¡Hasta luego!")
            break
            
        else:
            print("Opción no válida")


if __name__ == "__main__":
    main()


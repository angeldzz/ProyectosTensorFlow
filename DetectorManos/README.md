# 🤖 Detector de Gestos con Inteligencia Artificial

Un sistema completo de reconocimiento de gestos de mano en tiempo real utilizando MediaPipe, TensorFlow y OpenCV. Este proyecto permite detectar y clasificar diferentes gestos como piedra, papel, tijera, lagarto, bien y mal a través de la cámara web.

## 🌟 Características

- **Detección en tiempo real**: Reconocimiento instantáneo de gestos usando la cámara web
- **6 gestos soportados**: Piedra, papel, tijera, lagarto, bien y mal
- **Alta precisión**: Modelo de red neuronal entrenado con múltiples ejemplos
- **Interfaz visual**: Visualización de landmarks de manos y resultados en pantalla
- **Fácil extensión**: Sistema modular para agregar nuevos gestos

## 📋 Requisitos del Sistema

- Python 3.8 o superior
- Cámara web funcional
- Sistema operativo: Windows, macOS o Linux

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd DetectorManos
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### Dependencias principales
- **TensorFlow 2.19.0**: Framework de aprendizaje automático
- **MediaPipe 0.10.14**: Detección de manos y landmarks
- **OpenCV 4.12.0.88**: Procesamiento de imágenes y video
- **NumPy 2.1.3**: Manipulación de arrays numéricos
- **Scikit-learn 1.7.1**: Herramientas de machine learning

## 🚀 Uso del Sistema

### Ejecutar el detector
```bash
python DetectorGestosIA.py
```

### Controles
- **Muestra tu mano** frente a la cámara para ver el gesto detectado
- **Presiona 'q'** para salir del programa

## 📊 Entrenar el Modelo

### 1. Recolectar datos de gestos
```bash
python recolectorDeGestos.py
```

**Instrucciones para recolección:**
1. Modifica la variable `GESTO` en el archivo con el nombre del gesto a capturar
2. Ejecuta el script y posiciona tu mano frente a la cámara
3. Presiona **'s'** para guardar ejemplos del gesto (recomendado: 100-200 ejemplos)
4. Presiona **'q'** para finalizar la recolección
5. Repite el proceso para cada gesto

### 2. Entrenar el modelo
```bash
python entrenar_modelo.py
```

El entrenamiento generará el archivo `gestos_modelo_v2.keras` con el modelo entrenado.

## 📁 Estructura del Proyecto

```
DetectorManos/
├── DetectorGestosIA.py          # Detector principal en tiempo real
├── entrenar_modelo.py           # Script de entrenamiento del modelo
├── recolectorDeGestos.py        # Recolector de datos de gestos
├── gestos_modelo_v2.keras       # Modelo entrenado (generado)
├── datos/                       # Carpeta con datos de entrenamiento
│   ├── papel.csv
│   ├── tijera.csv
│   ├── piedra.csv
│   ├── lagarto.csv
│   ├── bien.csv
│   └── mal.csv
└── requirements.txt             # Dependencias del proyecto
```

## 🔧 Configuración del Modelo

### Arquitectura de la Red Neuronal
- **Capa de entrada**: 63 características (21 landmarks × 3 coordenadas)
- **Capa oculta 1**: 256 neuronas + ReLU + Dropout (0.4)
- **Capa oculta 2**: 128 neuronas + ReLU + Dropout (0.4)
- **Capa de salida**: 6 neuronas + Softmax

### Parámetros de entrenamiento
- **Optimizador**: Adam (learning rate: 0.0001)
- **Función de pérdida**: Sparse Categorical Crossentropy
- **Épocas**: 2000
- **Batch size**: 32
- **Validación**: 20% de los datos

## 🎯 Gestos Soportados

| Gesto | Descripción |
|-------|-------------|
| 👋 Papel | Mano abierta con dedos extendidos |
| ✌️ Tijera | Dedos índice y medio extendidos |
| ✊ Piedra | Puño cerrado |
| 🦎 Lagarto | Gesto tipo "lagarto" de piedra-papel-tijera-lagarto-spock |
| 👍 Bien | Pulgar hacia arriba |
| 👎 Mal | Pulgar hacia abajo |

## 🔄 Personalización

### Agregar nuevos gestos
1. Modifica la lista `gestos` en todos los archivos Python
2. Recolecta datos del nuevo gesto usando `recolectorDeGestos.py`
3. Reentrena el modelo con `entrenar_modelo.py`

### Ajustar precisión
- Aumenta el número de ejemplos por gesto (recomendado: 200-500)
- Modifica los parámetros del modelo en `entrenar_modelo.py`
- Ajusta el learning rate o añade más capas

## 🐛 Solución de Problemas

### El modelo no detecta correctamente
- **Solución**: Recolecta más datos de entrenamiento
- **Consejo**: Asegúrate de tener buena iluminación y fondo contrastante

### Error de cámara
- **Solución**: Verifica que la cámara no esté siendo usada por otra aplicación
- **Consejo**: Cambia el índice de cámara en `cv2.VideoCapture(0)` a `1` o `2`

### Rendimiento lento
- **Solución**: Reduce la resolución de la cámara o el número de épocas de entrenamiento
- **Consejo**: Usa una GPU compatible con TensorFlow para acelerar el entrenamiento

## 📈 Métricas de Rendimiento

El modelo actual alcanza aproximadamente:
- **Precisión de entrenamiento**: ~95%
- **Precisión de validación**: ~90%
- **FPS en detección**: 15-30 (dependiendo del hardware)

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si tienes ideas para mejoras:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

Desarrollado con ❤️ para la comunidad de IA y visión por computadora.

---

### 📞 Soporte

Si tienes preguntas o necesitas ayuda:
- Abre un issue en el repositorio
- Revisa la documentación de las bibliotecas utilizadas
- Consulta los ejemplos en el código

**¡Diviértete detectando gestos! 🎉**
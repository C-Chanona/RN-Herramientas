import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model('model.h5')

# Lista de etiquetas para todas las clases
class_labels = ["Llave Allen", "Cinta Metrica", "Pinza Corte", "Desarmador", 'Marro', 'Martillo', 'Taladro', 'Clavo', 'Tornillo', 'Cutter', 'Llave Perica', 'Llave Inglesa', 'Cerrucho']

# Configurar la cámara
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

def preprocess_image(frame, size_image=150):
    img_resized = cv2.resize(frame, (size_image, size_image))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.reshape(size_image, size_image, 1)
    img_gray = img_gray.astype('float32') / 255
    return img_gray

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir frame (final de la transmisión?). Saliendo ...")
        break
    
    # Preprocesar la imagen
    img_gray = preprocess_image(frame)
    
    # Hacer predicciones
    prediction = model.predict(np.array([img_gray]))
    predicted_class_index = np.argmax(prediction[0])
    label = class_labels[predicted_class_index]
    
    # Mostrar la predicción en el frame
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Mostrar el frame
    cv2.imshow('Camara en vivo', frame)
    
    # Salir si se presiona 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Cuando todo esté hecho, liberar la captura
cap.release()
cv2.destroyAllWindows()

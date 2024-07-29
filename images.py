import cv2
import os

def capture_images(object_name, save_dir, num_images=1600):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cap = cv2.VideoCapture(0)  # Iniciar la captura de video desde la cámara
    
    print(f"Capturando {num_images} imágenes de {object_name}. Presiona 's' para empezar y 'q' para salir.")
    
    img_count = 1300
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            img_name = os.path.join(save_dir, f"{object_name}_{img_count}.jpg")
            cv2.imwrite(img_name, frame)
            img_count += 1
            print(f"Imagen {img_count}/{num_images} guardada en {img_name}")
            
            if img_count >= num_images:
                break
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captura completada. Imágenes guardadas en {save_dir}")

if __name__ == "__main__":
    object_name = 'clavo'  # Nombre del objeto
    save_dir = './dataset/clavo/'  # Directorio donde se guardarán las imágenes
    
    capture_images(object_name, save_dir)

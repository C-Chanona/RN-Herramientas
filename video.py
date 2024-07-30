# Script para extraer frames de un video

import cv2
import os
import random
import progressbar

videos = [
    'cinta_metrica',
    'clavo',
    'cutter',
    'desarmador',
    'llave_allen',
    'llave_inglesa',
    'llave_perica',
    'marro',
    'martillo',
    'pinza_corte',
    'serrucho',
    'taladro',
    'tornillo'
]

# Clases que necesitan más imágenes
special_classes = ['llave_perica', 'llave_inglesa']

try:
    os.mkdir('dataset')
except FileExistsError:
    pass

for video in progressbar.progressbar(videos, prefix="Extracting frames: "):
    vidcap = cv2.VideoCapture("videos/" + video + '.mp4')
    success, image = vidcap.read()
    count = 0
    os.makedirs(f"./dataset/{video}", exist_ok=True)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    with progressbar.ProgressBar(max_value=total_frames, prefix=f"Extracting frames from {video}: ") as bar:
        while success:
            frame_path = f"dataset/{video}/frame{count}.jpg"
            cv2.imwrite(frame_path, image)  # save frame as JPEG file
            frames.append(frame_path)
            success, image = vidcap.read()
            count += 1
            bar.update(count)
    
    # Seleccionar más imágenes para las clases especiales
    if video in special_classes:
        selected_frames = random.sample(frames, min(len(frames), len(frames) // 2))
    else:
        selected_frames = random.sample(frames, (len(frames) // 2) // 2)
    
    # Eliminar las imágenes no seleccionadas
    for frame in frames:
        if frame not in selected_frames:
            os.remove(frame)
    
    print(f"{video}: {len(selected_frames)} imágenes seleccionadas")

# Imprimir el número final de imágenes por clase
print("\nNúmero final de imágenes por clase:")
for video in videos:
    num_images = len(os.listdir(f"./dataset/{video}"))
    print(f"{video}: {num_images} imágenes")
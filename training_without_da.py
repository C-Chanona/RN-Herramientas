import os
import cv2
import keras
import sklearn
import progressbar
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data(img_size=150):
    data = []
    target = []
    for index, class_ in enumerate(classes):
        folder_path = os.path.join('./dataset/', class_)
        print(f"normalizing {folder_path}")
        images = os.listdir(folder_path)
        with progressbar.ProgressBar(max_value=len(images), prefix=f"Processing {class_}: ") as bar:
            for i, img in enumerate(images):
                img_path = os.path.join(folder_path, img)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (img_size, img_size))
                    data.append(np.array(img))
                    target.append(index)
                except Exception as e:
                    print(f"Error reading file {img_path}: {e}")
                    continue
                bar.update(i + 1)
    data = np.array(data)
    data = data.reshape(data.shape[0], img_size, img_size, 1)
    target = np.array(target)
    new_target = keras.utils.to_categorical(target)
    return data, new_target

def model_train(data, target, num_classes):
    train_data, validation_data, train_target, validation_target = train_test_split(data, target, test_size=0.15, stratify=target)

    modelCNN = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(130, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Cambiado a softmax para clasificación multiclase
    ])

    modelCNN.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Cambiado a sparse_categorical_crossentropy
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True) # Detiene el entrenamiento si no hay mejora
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3) # Reduce la tasa de aprendizaje si el rendimiento se estanca

    history = modelCNN.fit(
        train_data,
        train_target,
        batch_size=32,
        epochs=30,
        verbose=1,
        validation_data=(validation_data, validation_target),
        callbacks=[early_stopping, reduce_lr]
    )

    modelCNN.save('model.h5')

    y_pred = modelCNN.predict(validation_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(validation_target, axis=1)
    confusion_mtx = sklearn.metrics.confusion_matrix(y_true, y_pred_classes)

    return history, confusion_mtx

def create_matrix(confusion_mtx):
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def loos_plot(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title('Gráfica de Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()


if __name__ == '__main__':
    classes = [
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

    num_classes = len(classes)
    data, target = load_data()
    history, confusion_mtx = model_train(data, target, num_classes)
    create_matrix(confusion_mtx)
    loos_plot(history)


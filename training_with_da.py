import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_images_from_directory(input_dir, label, size_image=150):
    data = []
    labels = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise Exception(f"No se pudo leer la imagen: {img_path}")
                
                img_resized = cv2.resize(img, (size_image, size_image))
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                img_gray = img_gray.reshape(size_image, size_image, 1)
                data.append(img_gray)
                labels.append(label)
            except Exception as e:
                print(f"Error al procesar la imagen {img_path}: {str(e)}")
    
    return np.array(data), np.array(labels)

def preprocess_data(size_image=150):
    input_dirs = [
        ('./dataset/llave_allen/', 0),
        ('./dataset/cinta_metrica/', 1),
        ('./dataset/pinza_corte/', 2), 
        ('./dataset/desarmador/', 3), 
        ('./dataset/marro/', 4), 
        ('./dataset/martillo/', 5), 
        ('./dataset/taladro/', 6),
        ('./dataset/clavo/', 7),
        ('./dataset/tornillo/', 8),
        ('./dataset/cutter/', 9),
        ('./dataset/llave_perica/', 10),
        ('./dataset/llave_inglesa/', 11),
        ('./dataset/cerrucho/', 12)
    ]

    X, y = [], []
    for input_dir, label in input_dirs:
        X_class, y_class = load_images_from_directory(input_dir, label, size_image)
        X.append(X_class)
        y.append(y_class)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    print("Número de imágenes por clase:")
    for i, (input_dir, label) in enumerate(input_dirs):
        print(f"Clase {i} ({os.path.basename(os.path.dirname(input_dir))}): {np.sum(y == label)} imágenes")


    X = X.astype(float) / 255  # Normalizar los valores de los pixeles
    return X, y

def print_class_distribution(y, class_labels):
    unique, counts = np.unique(y, return_counts=True)
    for i, (label, count) in enumerate(zip(unique, counts)):
        print(f"Clase {i} ({class_labels[i]}): {count} imágenes")

def get_class_weights(y):
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(enumerate(class_weights))

def model_train(X, y, num_classes, datagen):
    modelCNN = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Cambiado a softmax para clasificación multiclase
    ])

    modelCNN.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Cambiado a sparse_categorical_crossentropy
        metrics=['accuracy']
    )

    print(modelCNN.summary())

    # Usar validación estratificada
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y)

    # Calcular los pesos de clase
    class_weights = get_class_weights(y_train)
    print("Pesos de clase:", class_weights)

    # # Crear datasets
    # train_dataset = create_dataset(X_train, y_train, batch_size=32)
    # val_dataset = create_dataset(X_val, y_val, batch_size=32, is_training=False)

    # Generadores de datos
    train_generator = datagen.flow(X_train, y_train, batch_size=32)
    validation_generator = datagen.flow(X_val, y_val, batch_size=32)

    # Calcular los pasos por época correctamente
    steps_per_epoch = len(X_train) // 32
    validation_steps = len(X_val) // 32

    # Asegurarse de que los pasos sean al menos 1
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True) # Detiene el entrenamiento si no hay mejora
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3) # Reduce la tasa de aprendizaje si el rendimiento se estanca

    history = modelCNN.fit(
        train_generator,
        epochs=30,  # el early stopping detendrá si es necesario
        validation_data=validation_generator,
        steps_per_epoch=len(X_train) // 32,
        validation_steps=len(X_val) // 32,
        class_weight=class_weights,
        callbacks=[reduce_lr, early_stopping]
    )

    modelCNN.save('model.h5')

    return modelCNN, X_val, y_val, history


def data_increment(X):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=15,
        zoom_range=[0.7, 1.4],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    datagen.fit(X)

    return datagen

def create_dataset(X, y, batch_size, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Esto hará que el dataset se repita indefinidamente
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def generate_confusion_matrix(model, X_test, y_test, class_labels):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Obtén las clases únicas presentes en y_test y y_pred_classes
    np.unique(np.concatenate((y_test, y_pred_classes)))
    
    cm = confusion_matrix(y_test, y_pred_classes, labels=range(len(class_labels)))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Imprimir informe de clasificación
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred_classes, target_names=class_labels))

def plot_loss(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title('Gráfica de Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

if __name__ == "__main__":
    X, y = preprocess_data()
    datagen = data_increment(X)
    print(f"Tamaño total del conjunto de datos: {len(X)}")
    print(f"Número de clases: {len(np.unique(y))}")
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    
    class_labels = ["Llave Allen", "Cinta Metrica", "Pinza Corte", "Desarmador", 'Marro', 'Martillo', 'Taladro', 'Clavo', 'Tornillo', 'Cutter', 'Llave Perica', 'Llave Inglesa', 'Cerrucho']    
    num_classes = len(set(y))
    
    model, X_test, y_test, history = model_train(X, y, num_classes, datagen)
    generate_confusion_matrix(model, X_test, y_test, class_labels)
    plot_loss(history)
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

# Папки с данными
train_close_dir = 'C:/Users/USER/PythonProject1/.venv/Scripts/train/Closed_Eyes/train'
train_open_dir = 'C:/Users/USER/PythonProject1/.venv/Scripts/train/Open_Eyes/train'

val_close_dir = 'C:/Users/USER/PythonProject1/.venv/Scripts/train/Closed_Eyes/val'
val_open_dir = 'C:/Users/USER/PythonProject1/.venv/Scripts/train/Open_Eyes/val'

train_close = [os.path.join(train_close_dir, i) for i in os.listdir(train_close_dir)]
train_open = [os.path.join(train_open_dir, i) for i in os.listdir(train_open_dir)]
val_close = [os.path.join(val_close_dir, i) for i in os.listdir(val_close_dir)]
val_open = [os.path.join(val_open_dir, i) for i in os.listdir(val_open_dir)]

train_set = train_close + train_open
val_set = val_close + val_open


# Предобработка изображений
def preprocess_images(image_list, label, new_size):
    x, y = [], []
    for image in image_list:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img, new_size)
        img_norm = img_resize.astype(np.float32) / 255.0
        x.append(img_norm)
        y.append(label)
    return np.expand_dims(np.array(x), -1), np.array(y)


new_size = (64, 64)

X_train_closed, y_train_closed = preprocess_images(train_close, label=0, new_size=new_size)
X_train_open, y_train_open = preprocess_images(train_open, label=1, new_size=new_size)
X_val_closed, y_val_closed = preprocess_images(val_close, label=0, new_size=new_size)
X_val_open, y_val_open = preprocess_images(val_open, label=1, new_size=new_size)

X_train = np.concatenate([X_train_closed, X_train_open], axis=0)
y_train = np.concatenate([y_train_closed, y_train_open], axis=0)
X_val = np.concatenate([X_val_closed, X_val_open], axis=0)
y_val = np.concatenate([y_val_closed, y_val_open], axis=0)

y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

# Проверка данных
print("Distribution of labels in the training data:")
unique, counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
print(dict(zip(unique, counts)))

# Аугментация данных
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)


# Модель CNN
def create_cnn(input_shape=(64, 64, 1)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


cnn_model = create_cnn(input_shape=(64, 64, 1))
batch_size = 32
epochs = 10
# Обучение модели
history = cnn_model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    verbose=1
)

# Сохранение модели
cnn_model.save('cnn_eye_state_model.h5')

# Оценка модели
val_loss, val_accuracy = cnn_model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # График функции потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Построение графиков
plot_training_history(history)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Предсказания на валидационном наборе
y_val_true = np.argmax(y_val, axis=1)
y_val_pred = np.argmax(cnn_model.predict(X_val), axis=1)

# Вывод матрицы ошибок
plot_confusion_matrix(y_val_true, y_val_pred, class_names=["Closed", "Open"])

# --- Отчёт о классификации ---
print("\nClassification Report:\n")
print(classification_report(y_val_true, y_val_pred, target_names=["Closed", "Open"]))



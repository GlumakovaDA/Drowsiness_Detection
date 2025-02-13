import cv2
import numpy as np
from keras.models import load_model
import time
from playsound import playsound
import threading


model = load_model('C:/Users/USER/PythonProject1/.venv/Scripts/cnn_eye_state_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Параметры тревоги
required_duration = 5
start_time = None

# Размер входного изображения для модели
img_size = (64, 64)

def preprocess_eye(eye_img):
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)  # Преобразуем в оттенки серого
    eye_img = cv2.resize(eye_img, img_size)  # Приведение к размеру модели
    eye_img = eye_img / 255.0  # Нормализация
    eye_img = eye_img.reshape(1, *img_size, 1)  # Преобразуем в формат (1, H, W, C)
    return eye_img

def alarm():
    playsound('Radar.wav')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование изображения в градации серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Определение области лица
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Обнаружение глаз в области лица
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        closed_eyes_count = 0  # Счётчик закрытых глаз

        for (ex, ey, ew, eh) in eyes:
            eye = roi_color[ey:ey + eh, ex:ex + ew]
            preprocessed_eye = preprocess_eye(eye)

            # Предсказание состояния глаза
            prediction = model.predict(preprocessed_eye)
            eye_state = "Closed" if prediction[0][0] > 0.5 else "Open"

            # Рисуем рамки вокруг глаз
            color = (0, 0, 255) if eye_state == "Closed" else (0, 255, 0)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(frame, eye_state, (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Увеличиваем счётчик, если глаз закрыт
            if eye_state == "Closed":
                closed_eyes_count += 1

        # Проверка на сонливость: оба глаза должны быть закрыты
        if closed_eyes_count == 2 :
            if start_time is None:
                start_time = time.time()  # Запуск таймера
            elif time.time() - start_time >= required_duration:
                cv2.putText(frame, "Drowsiness Detected!", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if not alarm_active:
                    alarm_active = True
                    threading.Thread(target=alarm, daemon=True).start()
        else:
            start_time = None  # Сброс таймера, если глаза открыты
            alarm_active = False

    # Отображение изображения
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Нажмите 'q', чтобы выйти
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

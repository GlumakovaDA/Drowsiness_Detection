import cv2
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
import dlib
import time
from playsound import playsound
import threading

# Инициализация моделей
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')


# Функция для расчета EAR
def EAR(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])  # Верхняя и нижняя точки
    B = np.linalg.norm(eye_points[2] - eye_points[4])  # Другие верхняя и нижняя
    C = np.linalg.norm(eye_points[0] - eye_points[3])  # Расстояние между углами глаза
    ear = (A + B) / (2.0 * C)
    return ear

# Функция для извлечения координат глаз
def extract_eye_landmarks(shape, eye_indices):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in eye_indices], dtype=np.float32)

# Индексы точек глаза из 68-точечной модели
LEFT_EYE = [36, 37, 38, 39, 40, 41]  # Левый глаз
RIGHT_EYE = [42, 43, 44, 45, 46, 47]  # Правый глаз

required_duration = 5
start_time = None

def alarm():
    playsound('Radar.wav')
# Подключение камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        # Извлекаем маркеры лица
        landmarks = landmark_predictor(gray, face)

        # Извлекаем координаты глаз
        left_eye_points = extract_eye_landmarks(landmarks, LEFT_EYE)
        right_eye_points = extract_eye_landmarks(landmarks, RIGHT_EYE)

        # Расчет EAR для каждого глаза
        left_ear = EAR(left_eye_points)
        right_ear = EAR(right_eye_points)

        # Отображение точки глаза
        for (x, y) in left_eye_points:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
        for (x, y) in right_eye_points:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Вывод значений EAR
        cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Детектирование сонливости
        condition = left_ear < 0.25 and right_ear < 0.25

        if condition:
            if start_time is None:
                start_time = time.time()  # Запускаем таймер
            elif time.time() - start_time >= required_duration:
                cv2.putText(frame, "Drowsiness Detected!", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if not alarm_active:
                    alarm_active = True
                    threading.Thread(target=alarm, daemon=True).start()
        else:
            start_time = None
            alarm_active = False

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
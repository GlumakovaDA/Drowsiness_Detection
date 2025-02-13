<h1>Распознавание сонливости</h1><br>
Этот проект представляет собой систему распознавания сонливости с двумя вариантами реализации:<br>
Модель CNN: Использует сверточную нейронную сеть (CNN) для классификации состояния глаз (открыты/закрыты) по изображениям глаз.<br>
Метод EAR: Определяет уровень открытости глаз с помощью анализа маркеров лица (landmarks) и расчета коэффициента EAR (Eye Aspect Ratio).<br>
Обе системы могут использоваться в различных целях, к примеру для мониторинга водителя в реальном времени и оповещения при обнаружении признаков сонливости.<br>
<h2>Набор данных</h2><br>
Для модели, основанной на CNN использовался датасет, состоящий из 4000 фотографий открытых и закрытых глаз:<br>
https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset<br>
<h2>Модель, основанная на CNN</h2><br>
Файлы: CNN.py и CNNdetection.py<br>
<h3>Описание метода:</h3><br>
Использует сверточную нейронную сеть для классификации изображений глаз.<br>
Данные разделяются на тренировочную и валидационную выборки.<br>
Модель обучается на изображениях закрытых и открытых глаз.<br>
Реализована система тревожного сигнала при обнаружении сонливости.<br>
<h3>Используемые библиотеки:</h3><br>
tensorflow, keras — для создания и обучения CNN.<br>
opencv — для обработки изображений с камеры.<br>
numpy, matplotlib, seaborn — для обработки данных и визуализации.<br>
playsound, threading — для воспроизведения звукового сигнала.<br>
<br>
<h2>Модель, основанная на EAR</h2><br>
<h3>Описание метода:</h3><>br
Использует метод анализа маркеров лица (landmarks) для определения состояния глаз.<br>
Коэффициент EAR рассчитывается на основе расстояний между контрольными точками глаза.<br>
Если EAR опускается ниже заданного порога в течение нескольких секунд, активируется тревожный сигнал.<br>
<h3>Используемые библиотеки:</h3><br>
opencv — обработка изображений с веб-камеры.<br>
dlib — обнаружение лиц и извлечение маркеров лица.<br>
numpy — обработка координат глаз.<br>
playsound, threading — звуковое оповещение.<br>
<br>
Основано на статье Prateek Agrawal, Charu Gupta, Anand Sharma, Vishu Madaan, Nisheeth Joshi.
«Machine Learning and Data Science: Fundamentals and Applications», Scrivener Publishing
LLC, 2022.

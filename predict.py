# скрипт распознавания формы лекарственных средств

from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os


# загружаем модель
MODEL_NAME = 'drug_form_classifier.h5'
print("загрузка модели...")
model = keras.models.load_model(MODEL_NAME)
print("модель загружена!")

# загружаем названия классов
CLASS_NAMES = []
if os.path.exists('class_names.txt'):
    with open('class_names.txt', 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    print(f"загружено классов: {len(CLASS_NAMES)}")
else:
    print("файл class_names.txt не найден, используем значения по умолчанию")
    CLASS_NAMES = ['capsules', 'injections', 'ointment', 'suspension', 'tablets']

print()


def predict_drug_form(image_path):
    # распознает форму выпуска лекарства по изображению
    # параметры:
    #   image_path: путь к изображению
    # возвращает:
    #   tuple: (название класса, уверенность в %)
    print(f"анализ изображения: {image_path}")
    
    # загрузка и предобработка изображения
    img = image.load_img(image_path, target_size=(384, 384))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # нормализация
    
    print("изображение подготовлено!")
    
    # предсказание модели
    print("предсказание...")
    predictions = model.predict(img_array, verbose=0)
    
    # получение результата
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # получаем название предсказанного класса
    if predicted_class_index < len(CLASS_NAMES):
        predicted_class = CLASS_NAMES[predicted_class_index]
    else:
        predicted_class = f"Class_{predicted_class_index}"
    
    return predicted_class, confidence * 100

# использование

if __name__ == "__main__":
    # папка с тестовыми изображениями
    TEST_FOLDER = "test_images"
    
    # проверяем, существует ли папка
    if not os.path.exists(TEST_FOLDER):
        print(f"папка {TEST_FOLDER} не найдена!")
        print(f"создайте папку {TEST_FOLDER} и поместите туда тестовые изображения.")
        exit()
    
    # получаем список всех изображений в папке
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    test_images = []
    
    for filename in os.listdir(TEST_FOLDER):
        if any(filename.endswith(ext) for ext in image_extensions):
            test_images.append(os.path.join(TEST_FOLDER, filename))
    
    if not test_images:
        print(f"в папке {TEST_FOLDER} не найдено изображений!")
        exit()
    
    print(f"\nнайдено изображений для тестирования: {len(test_images)}")
    print("=" * 60)
    print()
    
    # обрабатываем каждое изображение
    for i, image_path in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] файл: {os.path.basename(image_path)}")
        print("-" * 60)
        
        try:
            # выполняем распознавание
            result_class, confidence = predict_drug_form(image_path)
            
            if result_class:
                print()
                print("результат:")
                print(f"  форма лекарства: {result_class.upper()}")
                print(f"  уверенность: {confidence:.2f}%")
                
                # интерпретация уверенности
                if confidence > 80:
                    print("  статус: высокая уверенность")
                elif confidence > 60:
                    print("  статус: средняя уверенность")
                else:
                    print("  статус: низкая уверенность")
            else:
                print("ошибка: не удалось распознать изображение")
        
        except Exception as e:
            print(f"ошибка обработки: {e}")
        
        print()
    
    print("=" * 60)
    print(f"обработка завершена! всего обработано: {len(test_images)} изображений")
    print("=" * 60)


from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# загружаем модель один раз при импорте модуля
MODEL = None
CLASS_NAMES = []

def load_model():
    # загрузка модели и названий классов (один раз)
    global MODEL, CLASS_NAMES
    
    if MODEL is None:
        if not os.path.exists('drug_form_classifier.h5'):
            print("ошибка: файл модели 'drug_form_classifier.h5' не найден!")
            return False
        
        print("загрузка модели...")
        MODEL = keras.models.load_model('drug_form_classifier.h5')
        print("модель загружена!")
        
        # загрузка названий классов
        if os.path.exists('class_names.txt'):
            with open('class_names.txt', 'r') as f:
                CLASS_NAMES = [line.strip() for line in f.readlines()]
        else:
            CLASS_NAMES = ['capsules', 'injections', 'ointment', 'suspension', 'tablets']
        
        print(f"загружено классов: {CLASS_NAMES}")
    
    return True

def predict_drug_form(image_path):
    # предсказание формы выпуска лекарства по изображению
    global MODEL, CLASS_NAMES
    
    # загружаем модель если ещё не загружена
    if not load_model():
        return None, None
    
    if not os.path.exists(image_path):
        print(f"ошибка: файл {image_path} не найден!")
        return None, None
    
    # подготовка изображений
    img = image.load_img(image_path, target_size=(384, 384))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # предсказание
    predictions = MODEL.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    predicted_class = CLASS_NAMES[predicted_class_index] if predicted_class_index < len(CLASS_NAMES) else f"Class_{predicted_class_index}"
    
    return predicted_class, confidence * 100

# тест всех изображений из папки test_images
if __name__ == "__main__":
    
    # папка с тестовыми изображениями
    TEST_FOLDER = "test_images"
    
    # проверяем существование папки
    if not os.path.exists(TEST_FOLDER):
        print(f"папка {TEST_FOLDER} не найдена!")
        print(f"создайте папку {TEST_FOLDER} и поместите туда тестовые изображения.")
        exit()
    
    # получаем список всех изображений
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    test_images = []
    
    for filename in os.listdir(TEST_FOLDER):
        if any(filename.endswith(ext) for ext in image_extensions):
            test_images.append(os.path.join(TEST_FOLDER, filename))
    
    if not test_images:
        print(f"в папке {TEST_FOLDER} не найдено изображений!")
        exit()
    
    print(f"\nнайдено {len(test_images)} изображений для тестирования")
    print("=" * 60)
    print()
    
    # обрабатываем каждое изображение
    for i, image_path in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] обработка: {os.path.basename(image_path)}")
        print("-" * 60)
        
        try:
            # получаем предсказание
            predicted_class, confidence = predict_drug_form(image_path)
            
            if predicted_class:
                print(f"\nпредсказание:")
                print(f"  форма лекарства: {predicted_class.upper()}")
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
    print(f"тестирование завершено! всего обработано: {len(test_images)} изображений")
    print("=" * 60)


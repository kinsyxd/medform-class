from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def predict_drug_form(image_path):
    # Load model
    if not os.path.exists('drug_form_classifier.h5'):
        return None, None
    
    model = keras.models.load_model('drug_form_classifier.h5')
    
    # Load class names
    CLASS_NAMES = []
    if os.path.exists('class_names.txt'):
        with open('class_names.txt', 'r') as f:
            CLASS_NAMES = [line.strip() for line in f.readlines()]
    else:
        CLASS_NAMES = ['capsules', 'injections', 'ointment', 'suspension', 'tablets']
    
    if not os.path.exists(image_path):
        print(f"ERROR: File {image_path} not found!")
        return None, None
    
    #подготовка изображений
    img = image.load_img(image_path, target_size=(384, 384))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    #предсказание
    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    predicted_class = CLASS_NAMES[predicted_class_index] if predicted_class_index < len(CLASS_NAMES) else f"Class_{predicted_class_index}"
    
    return predicted_class, confidence * 100

# Тест всех изображений из папки test_images
if __name__ == "__main__":
    
    # Папка с тестовыми изображениями
    TEST_FOLDER = "test_images"
    
    # Проверяем существование папки
    if not os.path.exists(TEST_FOLDER):
        print(f"Folder {TEST_FOLDER} not found!")
        print(f"Create folder {TEST_FOLDER} and place test images there.")
        exit()
    
    # Получаем список всех изображений
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    test_images = []
    
    for filename in os.listdir(TEST_FOLDER):
        if any(filename.endswith(ext) for ext in image_extensions):
            test_images.append(os.path.join(TEST_FOLDER, filename))
    
    if not test_images:
        print(f"No images found in folder {TEST_FOLDER}!")
        exit()
    
    print(f"\nFound {len(test_images)} images for testing")
    print("=" * 60)
    print()
    
    # Обрабатываем каждое изображение
    for i, image_path in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] Processing: {os.path.basename(image_path)}")
        print("-" * 60)
        
        try:
            # Получаем предсказание
            predicted_class, confidence = predict_drug_form(image_path)
            
            if predicted_class:
                print(f"\nPREDICTION:")
                print(f"  Drug form: {predicted_class.upper()}")
                print(f"  Confidence: {confidence:.2f}%")
                
                # Интерпретация уверенности
                if confidence > 80:
                    print("  Status: High confidence")
                elif confidence > 60:
                    print("  Status: Medium confidence")
                else:
                    print("  Status: Low confidence")
            else:
                print("Error: Failed to recognize image")
        
        except Exception as e:
            print(f"Error processing: {e}")
        
        print()
    
    print("=" * 60)
    print(f"Testing completed! Total processed: {len(test_images)} images")
    print("=" * 60)


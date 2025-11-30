# скрипт для детальной оценки качества модели
# выводит: матрицу ошибок, precision, recall, f1-score, примеры ошибок

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# ============================================================================
# настройки
# ============================================================================

MODEL_PATH = 'drug_form_classifier.h5'
DATA_DIR = 'dataset/train'
IMG_SIZE = 384
BATCH_SIZE = 16

# доля данных для тестирования (эти данные не использовались при обучении)
TEST_SPLIT = 0.2

# ============================================================================
# загрузка модели и данных
# ============================================================================

print("=" * 70)
print("ДЕТАЛЬНАЯ ОЦЕНКА МОДЕЛИ")
print("=" * 70)

# проверка наличия модели
if not os.path.exists(MODEL_PATH):
    print(f"ошибка: файл модели '{MODEL_PATH}' не найден!")
    print("сначала запустите train_model.py для обучения модели")
    exit()

# загрузка модели
print("\nзагрузка модели...")
model = keras.models.load_model(MODEL_PATH)
print("модель загружена!")

# загрузка названий классов
if os.path.exists('class_names.txt'):
    with open('class_names.txt', 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
else:
    CLASS_NAMES = sorted([f for f in os.listdir(DATA_DIR) 
                          if os.path.isdir(os.path.join(DATA_DIR, f))])

print(f"классы: {CLASS_NAMES}")
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# подготовка тестовых данных
# ============================================================================

# генератор без аугментации (только нормализация)
# используем тот же validation_split что и при обучении
test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=TEST_SPLIT
)

# загружаем тестовые данные (те же 20% что использовались для валидации)
# shuffle=False важен для корректного сопоставления предсказаний с метками
test_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

print(f"\nтестовых изображений: {test_generator.samples}")

# ============================================================================
# получение предсказаний
# ============================================================================

print("\nполучение предсказаний...")

# предсказания модели (вероятности для каждого класса)
predictions = model.predict(test_generator, verbose=1)

# индексы предсказанных классов
predicted_classes = np.argmax(predictions, axis=1)

# истинные метки классов
true_classes = test_generator.classes

# уверенность модели в предсказаниях
confidences = np.max(predictions, axis=1)

# ============================================================================
# расчёт метрик
# ============================================================================

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
print("=" * 70)

# общая точность
accuracy = np.mean(predicted_classes == true_classes)
print(f"\nОБЩАЯ ТОЧНОСТЬ: {accuracy * 100:.2f}%")

# матрица ошибок (confusion matrix)
print("\n" + "-" * 70)
print("МАТРИЦА ОШИБОК (строки - истинные классы, столбцы - предсказанные)")
print("-" * 70)

# создаём матрицу ошибок вручную (без sklearn)
confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for true_label, pred_label in zip(true_classes, predicted_classes):
    confusion_matrix[true_label][pred_label] += 1

# выводим заголовок таблицы
header = "истинный\\предск"
for name in CLASS_NAMES:
    header += f" | {name[:8]:>8}"
print(header)
print("-" * len(header))

# выводим строки матрицы
for i, row in enumerate(confusion_matrix):
    line = f"{CLASS_NAMES[i][:15]:<15}"
    for val in row:
        line += f" | {val:>8}"
    print(line)

# ============================================================================
# детальные метрики для каждого класса
# ============================================================================

print("\n" + "-" * 70)
print("МЕТРИКИ ПО КЛАССАМ")
print("-" * 70)
print(f"{'класс':<15} | {'precision':>10} | {'recall':>10} | {'f1-score':>10} | {'кол-во':>8}")
print("-" * 70)

# расчёт precision, recall, f1 для каждого класса
precisions = []
recalls = []
f1_scores = []
supports = []

for i in range(NUM_CLASSES):
    # true positives - правильно предсказанные как класс i
    tp = confusion_matrix[i][i]
    
    # false positives - неправильно предсказанные как класс i
    fp = sum(confusion_matrix[j][i] for j in range(NUM_CLASSES)) - tp
    
    # false negatives - класс i предсказан как другой класс
    fn = sum(confusion_matrix[i]) - tp
    
    # количество примеров класса
    support = sum(confusion_matrix[i])
    
    # precision = tp / (tp + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # recall = tp / (tp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # f1 = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    supports.append(support)
    
    print(f"{CLASS_NAMES[i]:<15} | {precision*100:>9.2f}% | {recall*100:>9.2f}% | {f1*100:>9.2f}% | {support:>8}")

# средневзвешенные метрики
total_samples = sum(supports)
weighted_precision = sum(p * s for p, s in zip(precisions, supports)) / total_samples
weighted_recall = sum(r * s for r, s in zip(recalls, supports)) / total_samples
weighted_f1 = sum(f * s for f, s in zip(f1_scores, supports)) / total_samples

print("-" * 70)
print(f"{'СРЕДНЕВЗВЕШ.':<15} | {weighted_precision*100:>9.2f}% | {weighted_recall*100:>9.2f}% | {weighted_f1*100:>9.2f}% | {total_samples:>8}")

# ============================================================================
# анализ ошибок
# ============================================================================

print("\n" + "-" * 70)
print("ПРИМЕРЫ ОШИБОЧНЫХ ПРЕДСКАЗАНИЙ")
print("-" * 70)

# получаем пути к файлам
filenames = test_generator.filenames

# находим индексы ошибочных предсказаний
error_indices = np.where(predicted_classes != true_classes)[0]
num_errors = len(error_indices)

print(f"\nвсего ошибок: {num_errors} из {len(true_classes)} ({num_errors/len(true_classes)*100:.1f}%)")

if num_errors > 0:
    print(f"\nпервые {min(10, num_errors)} ошибок:\n")
    
    for idx in error_indices[:10]:
        true_class = CLASS_NAMES[true_classes[idx]]
        pred_class = CLASS_NAMES[predicted_classes[idx]]
        conf = confidences[idx] * 100
        filename = filenames[idx]
        
        print(f"  файл: {filename}")
        print(f"    истинный класс: {true_class}")
        print(f"    предсказано: {pred_class} (уверенность: {conf:.1f}%)")
        print()

# ============================================================================
# статистика по уверенности
# ============================================================================

print("-" * 70)
print("СТАТИСТИКА УВЕРЕННОСТИ МОДЕЛИ")
print("-" * 70)

# уверенность для правильных и неправильных предсказаний
correct_mask = predicted_classes == true_classes
correct_conf = confidences[correct_mask]
incorrect_conf = confidences[~correct_mask]

print(f"\nправильные предсказания:")
print(f"  средняя уверенность: {np.mean(correct_conf)*100:.1f}%")
print(f"  мин. уверенность: {np.min(correct_conf)*100:.1f}%")
print(f"  макс. уверенность: {np.max(correct_conf)*100:.1f}%")

if len(incorrect_conf) > 0:
    print(f"\nнеправильные предсказания:")
    print(f"  средняя уверенность: {np.mean(incorrect_conf)*100:.1f}%")
    print(f"  мин. уверенность: {np.min(incorrect_conf)*100:.1f}%")
    print(f"  макс. уверенность: {np.max(incorrect_conf)*100:.1f}%")

# распределение уверенности
print("\nраспределение уверенности:")
for threshold in [50, 60, 70, 80, 90]:
    count = np.sum(confidences >= threshold/100)
    print(f"  >= {threshold}%: {count} изображений ({count/len(confidences)*100:.1f}%)")

# ============================================================================
# итоги
# ============================================================================

print("\n" + "=" * 70)
print("ИТОГИ")
print("=" * 70)
print(f"\nточность модели: {accuracy * 100:.2f}%")
print(f"f1-score (взвешенный): {weighted_f1 * 100:.2f}%")

if accuracy >= 0.9:
    print("\nоценка: ОТЛИЧНО - модель работает хорошо!")
elif accuracy >= 0.7:
    print("\nоценка: ХОРОШО - модель работает приемлемо")
elif accuracy >= 0.5:
    print("\nоценка: УДОВЛЕТВОРИТЕЛЬНО - требуется улучшение")
else:
    print("\nоценка: ПЛОХО - модель требует серьёзной доработки")

# рекомендации на основе анализа
print("\nрекомендации:")

# находим классы с низким recall (модель плохо их распознаёт)
for i, recall in enumerate(recalls):
    if recall < 0.7:
        print(f"  - класс '{CLASS_NAMES[i]}' имеет низкий recall ({recall*100:.1f}%) - добавьте больше примеров")

# находим классы которые часто путаются
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and confusion_matrix[i][j] > 2:
            print(f"  - класс '{CLASS_NAMES[i]}' часто путается с '{CLASS_NAMES[j]}' ({confusion_matrix[i][j]} раз)")

print("\n" + "=" * 70)
print("оценка завершена!")
print("=" * 70)


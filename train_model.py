# импорт библиотек
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ============================================================================
# НАСТРОЙКИ ПРОЕКТА
# ============================================================================

# путь к папке с датасетом
TRAIN_DIR = 'dataset/train'

# размер изображений
IMG_SIZE = 384

# Количество изображений, обрабатываемых за раз
BATCH_SIZE = 16  # Меньше = меньше памяти, но медленнее

# Количество повторений всего датасета
EPOCHS = 25

# Имя файла для сохранения модели
MODEL_NAME = 'drug_form_classifier.h5'

print(f"\nSettings:")
print(f"- Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Epochs: {EPOCHS}")
print(f"- Directory: {TRAIN_DIR}\n")


# определяем количество классов (в будущем возможно добавлю разделение сиропов и суспензий)
NUM_CLASSES = len([f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))])
print(f"Found classes: {NUM_CLASSES}")

# получаем названия классов по нзваниям папок
CLASS_NAMES = sorted([f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))])
print(f"Classes: {CLASS_NAMES}")

# генератор для аугментации данных
train_datagen = ImageDataGenerator(
    rescale=1./255,              # пиксели от 0 до 1
    rotation_range=15,           # поворот до +-15 градусов
    width_shift_range=0.1,      # сдвиг по ширине на 10%
    height_shift_range=0.1,     # сдвиг по высоте на 10%
    horizontal_flip=True,        # горизонтальное отражение
    zoom_range=0.1,              # увеличение/уменьшение на 10%
    brightness_range=[0.8, 1.2], # изменение яркости
    fill_mode='nearest'          # заполнение пустых областей
)

# Загружаем изображения из папок
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),  # Изменяем размер всех изображений
    batch_size=BATCH_SIZE,              # сколько изображений за раз?
    class_mode='categorical',           # категориальная классификация
    shuffle=True                        # Перемешиваем данные
)

print(f"Loaded images: {train_generator.samples}")
print(f"Classes: {train_generator.num_classes}")
print()

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

print("Base model: MobileNetV2")

# замораживаем модель mobilenetv2, т.к. мы должны обучать только новые слои
base_model.trainable = False

# создаем полную модель
model = keras.Sequential([
    base_model,                           # базовая модель MobileNetV2
    GlobalAveragePooling2D(),             # усреднение данных
    Dense(256, activation='relu'),        # полносвязный слой (256 нейронов)
    Dropout(0.5),                         # отключаем 50% нейронов (защита от переобучения)
    Dense(NUM_CLASSES, activation='softmax')  # выходной слой (5 классов: capsules, injections, ointment, suspension, tablets)
])

# компиляция
model.compile(
    optimizer='adam',                      # Алгоритм оптимизации
    loss='categorical_crossentropy',      # Функция потерь
    metrics=['accuracy']                   # Метрика качества
)

# выводим структуру модели
print("Model summary:")
model.summary()

# ШАГ 3: callback'и для мониторинга и сохранения лучших вариантов/весов модели

# колбэк для сохранения лучшей модели
checkpoint = ModelCheckpoint(
    MODEL_NAME,
    monitor='accuracy',           # мониторинг точности
    save_best_only=True,          # перезапись модели ТОЛЬОК при улучшении метрик
    verbose=1                     # вербоз 1 показывает шкалу и метрики эпох
)

# колбэк для ранней остановки (если модель не улучшается)
early_stop = EarlyStopping(
    monitor='accuracy',
    patience=3,                   # ожидание пяти эпох без улучшения
    restore_best_weights=True     # восстанавливаются только лучшие веса
)


# процесс обучения
print("="*20)
print("Training started")
print("="*20)

# обучаем модель
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

print()
print("="*20)
print("Training completed")
print("="*20)

# расчёт финальной точности
final_accuracy = history.history['accuracy'][-1]
print(f"\nFinal accuracy: {final_accuracy * 100:.3f}%")

if final_accuracy > 0.9:
    print("Excellent result!")
elif final_accuracy > 0.6:
    print("Good result")
else:
    print("Poor result - needs improvement")

# сохранение названий классов которые использовались для тренировки
with open('class_names.txt', 'w') as f:
    for name in CLASS_NAMES:
        f.write(name + '\n')

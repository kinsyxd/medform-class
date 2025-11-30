# импорт библиотек
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ============================================================================
# настройки проекта
# ============================================================================

# путь к папке с датасетом
TRAIN_DIR = 'dataset/train'

# размер изображений
IMG_SIZE = 384

# количество изображений, обрабатываемых за раз
BATCH_SIZE = 16  # меньше = меньше памяти, но медленнее

# количество повторений всего датасета
EPOCHS = 25

# имя файла для сохранения модели
MODEL_NAME = 'drug_form_classifier.h5'

print(f"\nнастройки:")
print(f"- размер изображения: {IMG_SIZE}x{IMG_SIZE}")
print(f"- размер пакета: {BATCH_SIZE}")
print(f"- эпохи: {EPOCHS}")
print(f"- каталог: {TRAIN_DIR}\n")


# определяем количество классов (в будущем возможно добавлю разделение сиропов и суспензий)
NUM_CLASSES = len([f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))])
print(f"найдено классов: {NUM_CLASSES}")

# получаем названия классов по названиям папок
CLASS_NAMES = sorted([f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))])
print(f"классы: {CLASS_NAMES}")

# генератор для аугментации данных (с разделением на train/validation)
train_datagen = ImageDataGenerator(
    rescale=1./255,              # пиксели от 0 до 1
    rotation_range=15,           # поворот до +-15 градусов
    width_shift_range=0.1,       # сдвиг по ширине на 10%
    height_shift_range=0.1,      # сдвиг по высоте на 10%
    horizontal_flip=True,        # горизонтальное отражение
    zoom_range=0.1,              # увеличение/уменьшение на 10%
    brightness_range=[0.8, 1.2], # изменение яркости
    fill_mode='nearest',         # заполнение пустых областей
    validation_split=0.2         # 20% данных для валидации
)

# загружаем тренировочные изображения (80%)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),  # изменяем размер всех изображений
    batch_size=BATCH_SIZE,              # сколько изображений за раз?
    class_mode='categorical',           # категориальная классификация
    shuffle=True,                       # перемешиваем данные
    subset='training'                   # только тренировочные данные
)

# загружаем валидационные изображения (20%)
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,                      # не перемешиваем для валидации
    subset='validation'                 # только валидационные данные
)

print(f"тренировочных изображений: {train_generator.samples}")
print(f"валидационных изображений: {validation_generator.samples}")
print(f"классы: {train_generator.num_classes}")
print()

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

print("базовая модель: MobileNetV2")

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
    optimizer='adam',                      # алгоритм оптимизации
    loss='categorical_crossentropy',      # функция потерь
    metrics=['accuracy']                   # метрика качества
)

# выводим структуру модели
print("структура модели:")
model.summary()

# шаг 3: callback'и для мониторинга и сохранения лучших вариантов/весов модели

# колбэк для сохранения лучшей модели
checkpoint = ModelCheckpoint(
    MODEL_NAME,
    monitor='val_accuracy',       # мониторинг валидационной точности
    save_best_only=True,          # перезапись модели только при улучшении метрик
    verbose=1                     # вербоз 1 показывает шкалу и метрики эпох
)

# колбэк для ранней остановки (если модель не улучшается)
early_stop = EarlyStopping(
    monitor='val_accuracy',       # мониторинг валидационной точности
    patience=5,                   # ожидание пяти эпох без улучшения
    restore_best_weights=True     # восстанавливаются только лучшие веса
)


# процесс обучения
print("="*20)
print("начало обучения")
print("="*20)

# обучаем модель
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,  # валидационные данные
    callbacks=[checkpoint, early_stop],
    verbose=1
)

print()
print("="*20)
print("обучение завершено")
print("="*20)

# расчёт финальной точности
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"\nфинальная точность обучения: {final_train_accuracy * 100:.3f}%")
print(f"финальная точность валидации: {final_val_accuracy * 100:.3f}%")

if final_val_accuracy > 0.9:
    print("отличный результат!")
elif final_val_accuracy > 0.6:
    print("хороший результат")
else:
    print("плохой результат - требует улучшений")

# сохранение названий классов которые использовались для тренировки
with open('class_names.txt', 'w') as f:
    for name in CLASS_NAMES:
        f.write(name + '\n')

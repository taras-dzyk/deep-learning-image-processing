def search():
    import pathlib
    data_dir = pathlib.Path('input').with_suffix('')
    image_count = len(list(data_dir.glob('**/*.jpg')))
    print(f'Found {image_count} images')

    print("Searching for crocodiles")
    import tensorflow as tf
    from tensorflow import keras
    model = tf.keras.models.load_model("model.keras")


    min_height = 197
    min_width = 300

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        #labels='inferred',                # Automatically infer the label from the subfolder
        label_mode='int',
        image_size=(min_height, min_width),# Resize all images
        batch_size=85,
        shuffle=True
    )


    print(f'Тестовий датасет: {test_ds.reduce(0, lambda x, _: x + 1)}')

    # зберемо датасети в батчі
    test_ds = test_ds.batch(50)



    import numpy as np
    from sklearn.metrics import classification_report

    true_labels = []
    predicted_labels = []
    predicted_proba = []
    predictions_full = []


    for images, labels in test_ds:
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to class indices
        predicted_probabilities = np.max(predictions, axis=1)

        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted_classes)
        predicted_proba.extend(predicted_probabilities)
        predictions_full.extend(predictions)
    report = classification_report(true_labels, predicted_labels)

    print(report)

def search_and_copy():
    import tensorflow as tf
    import numpy as np
    import pathlib
    import shutil
    from PIL import Image
    import os

    # Параметри
    #min_height = 197
    #min_width = 300
    min_height = 224
    min_width = 224
    data_dir = pathlib.Path('input')
    output_dir = pathlib.Path('output')
    output_dir.mkdir(exist_ok=True)
    image_size = (min_height, min_width)  # наприклад (224, 224)

    # Завантаження моделі
    model = tf.keras.models.load_model("model_binary.keras")

    # Проходимо по всіх зображеннях у підкаталогах
    image_paths = list(data_dir.glob("**/*.*"))  # Знаходимо всі файли в підпапках

    for image_path in image_paths:
        try:
            # Відкриваємо та обробляємо зображення
            img = Image.open(image_path).convert("RGB").resize((min_width, min_height))
            img_array = np.array(img) / 255.0  # нормалізуємо, якщо модель навчена на таких
            img_tensor = tf.expand_dims(img_array, axis=0)  # додаємо batch dimension

            # Прогноз класу
            prediction = model.predict(img_tensor)
            predicted_class = np.argmax(prediction, axis=1)[0]

            print(f'Predicted: {predicted_class}')

            # Якщо це клас 0 — копіюємо у вихідну папку
            if predicted_class == 1:
                target_path = output_dir / image_path.name
                shutil.copy(image_path, target_path)

        except Exception as e:
            print(f"Помилка з файлом {image_path}: {e}")


def search_and_copy_v2(probability=0.5):
    import os
    import shutil
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    from pathlib import Path

    # Завантаження моделі
    model_binary = tf.keras.models.load_model("model_binary.keras")
    
    # Папки
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Підтримувані розширення
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    # Розмір зображення має збігатися з input_shape моделі
    target_size = (224, 224)

    def is_image_file(path):
        return path.suffix.lower() in image_extensions

    def load_and_preprocess_image(image_path):
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img)
        return img_array

    # Проходимо всі зображення у 'input'
    for img_path in input_dir.glob("*.*"):
        if not img_path.is_file() or not is_image_file(img_path):
            printy(f"Пропущено: {img_path.name}")
            continue

        try:
            img_array = load_and_preprocess_image(img_path)
            img_tensor = tf.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
            pred = model_binary.predict(img_tensor, verbose=0)[0][0]

            if pred > probability:
                shutil.copy(img_path, output_dir / img_path.name)
                printg(f"Тримай крокодила! {img_path.name} (score: {pred:.3f})")
            else:
                printv(f"Не схоже: {img_path.name} (score: {pred:.3f})")

        except Exception as e:
            printr(f"Помилка з файлом {img_path.name}: {e}")

def printv(text):
    print(f"\033[95m{text}\033[0m")

def printg(text):
    print(f"\033[92m{text}\033[0m")

def printr(text):
    print(f"\033[91m{text}\033[0m")

def printy(text):
    print(f"\033[93m{text}\033[0m")

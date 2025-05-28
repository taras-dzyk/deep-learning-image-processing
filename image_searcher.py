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
    min_height = 197
    min_width = 300
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
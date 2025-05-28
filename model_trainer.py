
def train_model():
    import tensorflow as tf
    from tensorflow import keras

    import pathlib

    data_dir = pathlib.Path('dataset').with_suffix('')


    print(f'Preparing a model')
    # Load dataset

    # я модифікував це для resnet котрий вимагає мін 197×197
    min_height = 197
    min_width = 300

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',                # Automatically infer the label from the subfolder
        label_mode='int',
        image_size=(min_height, min_width),# Resize all images
        batch_size=85,
        shuffle=True
    )
    # Зберігаємо мапінг перед .unbatch()
    class_names = dataset.class_names

    # Тепер можна unbatch
    dataset = dataset.unbatch()

    # Підрахунок загальної кількості зображень
    total_count = sum(1 for _ in dataset)  # cardinality повертає tf.data.INFINITE_CARDINALITY або точну кількість

    # Розрахунок кількості для кожної вибірки
    train_size = int(0.7 * total_count)
    val_size = int(0.15 * total_count)
    test_size = total_count - train_size - val_size

    # Надрукувати мапінг: назва класу → індекс
    for idx, class_name in enumerate(class_names):
        print(f"{idx}: {class_name}")

    # Розбиття датасету
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size)

    print(f'Тренувальний датасет: {train_ds.reduce(0, lambda x, _: x + 1)}')
    print(f'Валідаційний датасет: {val_ds.reduce(0, lambda x, _: x + 1)}')
    print(f'Тестовий датасет: {test_ds.reduce(0, lambda x, _: x + 1)}')

    # зберемо датасети в батчі
    train_ds = train_ds.batch(50)
    val_ds = val_ds.batch(50)
    test_ds = test_ds.batch(50)





    import tensorflow as tf

    inputs = tf.keras.Input(shape=(197, 300, 3))

    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
        )

    history = model.fit(
        train_ds,
        epochs=3,
        validation_data=val_ds)


    model.summary()








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

    model.save("model.keras")



def train_one_vs_all():
    import tensorflow as tf

    print(f'Preparing a model')
    # Load dataset

        
    import pathlib

    data_dir = pathlib.Path('dataset').with_suffix('')

    # я модифікував це для resnet котрий вимагає мін 197×197
    min_height = 197
    min_width = 300

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        #labels='inferred',                # Automatically infer the label from the subfolder
        label_mode='int',
        image_size=(min_height, min_width),# Resize all images
        batch_size=50,
        shuffle=True
    )

    # Перетворимо багатокласові мітки у бінарні: 1 якщо "crocodiles", інакше 0
    class_names = dataset.class_names  # список папок
    print("Мапінг класів:", class_names)  # Важливо: перевірити, під яким номером "crocodiles"

    # Тепер можна unbatch
    dataset = dataset.unbatch()

    crocodile_index = class_names.index("crocodiles")

    def to_binary_label(image, label):
        binary_label = tf.cast(tf.equal(label, crocodile_index), tf.int32)
        return image, binary_label

    dataset = dataset.map(to_binary_label)




    # Підрахунок загальної кількості зображень
    total_count = sum(1 for _ in dataset)  # cardinality повертає tf.data.INFINITE_CARDINALITY або точну кількість

    # Розрахунок кількості для кожної вибірки
    train_size = int(0.7 * total_count)
    val_size = int(0.15 * total_count)
    test_size = total_count - train_size - val_size

    # Надрукувати мапінг: назва класу → індекс
    for idx, class_name in enumerate(class_names):
        print(f"{idx}: {class_name}")

    # Розбиття датасету
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size)

    print(f'Тренувальний датасет: {train_ds.reduce(0, lambda x, _: x + 1)}')
    print(f'Валідаційний датасет: {val_ds.reduce(0, lambda x, _: x + 1)}')
    print(f'Тестовий датасет: {test_ds.reduce(0, lambda x, _: x + 1)}')

    # зберемо датасети в батчі
    train_ds = train_ds.batch(50)
    val_ds = val_ds.batch(50)
    test_ds = test_ds.batch(50)


   

    inputs = tf.keras.Input(shape=(197, 300, 3))

    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model_binary = tf.keras.Model(inputs=inputs, outputs=outputs)

    model_binary.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    class_weights = {0: 1.0, 1: 2.0} #  клас 1 важливіший у 10 разів
    # Тренуємо
    history = model_binary.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight = class_weights
    )


    model_binary.summary()








    import numpy as np
    from sklearn.metrics import classification_report

    def evaluate_binary_model(model, dataset):
        y_true = []
        y_pred = []

        for batch in dataset:
            x_batch, y_batch = batch
            preds = model.predict(x_batch, verbose=0)

            # Збереження істинних міток та передбачень
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.squeeze())  # sigmoid -> ймовірності

        # Перетворюємо в бінарні мітки
        y_true_binary = np.array(y_true).astype(int)
        y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

        # Друк звіту
        print(classification_report(y_true_binary, y_pred_binary, digits=3))

    # Виклик для валід. датасету:
    evaluate_binary_model(model_binary, val_ds)

    model_binary.save("model_binary.keras")



def train_one_vs_all_v2():
    print(f'Preparing a model')

    import tensorflow as tf
    import pathlib

    data_dir = pathlib.Path('dataset')  # твоя основна папка з підпапками

    # Створюємо датасет з усіх зображень
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='int',
        image_size=(224, 224),     # змінюй на свій розмір
        batch_size=None,           # unbatched одразу
        shuffle=True
    )

    # Перетворимо багатокласові мітки у бінарні: 1 якщо "crocodiles", інакше 0
    class_names = dataset.class_names  # список папок
    print("Мапінг класів:", class_names)  # Важливо: перевірити, під яким номером "crocodiles"

    crocodile_index = class_names.index("crocodiles")

    def to_binary_label(image, label):
        binary_label = tf.cast(tf.equal(label, crocodile_index), tf.int32)
        return image, binary_label

    dataset_binary = dataset.map(to_binary_label)


    # --------------------
    # Розбиття на train/val
    # --------------------
    # Перемішуємо та кешуємо для кращої продуктивності
    dataset_binary = dataset_binary.shuffle(1000).cache()

    # Підраховуємо кількість прикладів
    dataset_size = dataset_binary.cardinality().numpy()
    train_size = int(0.8 * dataset_size)

    train_ds = dataset_binary.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset_binary.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)




    # --------------------
    # Побудова бінарної моделі
    # --------------------
    import tensorflow as tf
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='models/weights.h5'
    )
    base_model.trainable = False  # заморожуємо базову модель

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model_binary = tf.keras.Model(inputs, outputs)

    model_binary.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    model_binary.summary()


    # --------------------
    # Навчання
    # --------------------
    # За потреби: підрахунок ваг класів (бо крокодилів менше)
    neg, pos = 0, 0
    for _, label in dataset_binary:
        if label.numpy() == 1:
            pos += 1
        else:
            neg += 1

    total = neg + pos
    weight_for_0 = total / (2 * neg)
    weight_for_1 = total / (2 * pos)

    class_weights = {0: weight_for_0, 1: weight_for_1}
    print("Class weights:", class_weights)

    # Тренування
    model_binary.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights
    )


    # --------------------
    # Оцінка
    # --------------------
    import numpy as np
    from sklearn.metrics import classification_report

    def evaluate_binary_model(model, dataset):
        y_true = []
        y_pred = []

        for batch in dataset:
            x_batch, y_batch = batch
            preds = model.predict(x_batch, verbose=0)

            # Збереження істинних міток та передбачень
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.squeeze())  # sigmoid -> ймовірності

        # Перетворюємо в бінарні мітки
        y_true_binary = np.array(y_true).astype(int)
        y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
        print(f"Score: {y_pred}")


        # Друк звіту
        print(classification_report(y_true_binary, y_pred_binary, digits=3))

    # Виклик для валід. датасету:
    evaluate_binary_model(model_binary, val_ds)

    model_binary.save("model_binary.keras")
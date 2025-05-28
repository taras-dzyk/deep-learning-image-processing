

def search_and_copy_v2(probability=0.5):
    import os
    import shutil
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    from pathlib import Path

    model_binary = tf.keras.models.load_model("model_binary.keras")
    
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)


    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    target_size = (224, 224)

    def is_image_file(path):
        return path.suffix.lower() in image_extensions

    def load_and_preprocess_image(image_path):
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img)
        return img_array

   
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

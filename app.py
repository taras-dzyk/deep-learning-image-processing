from model_trainer import train_model, train_one_vs_all, train_one_vs_all_v2
from image_searcher import search, search_and_copy, search_and_copy_v2

def print_crocodile():
    import os
    import random

    folder = "_ascii"

    # Зібрати всі файли (наприклад, з розширенням .txt)
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith(".txt")]

    if not files:
        exit()

    random_file = random.choice(files)
    file_path = os.path.join(folder, random_file)

    with open(file_path, 'r', encoding='utf-8') as f:
        ascii_art = f.read()
    print(ascii_art)



print_crocodile()
print("Welcome to crocodile finder")


user_input = input(f'Do you want to train a model of find crocodiles? (type \'t\' for train, hit return for crocodile search)').strip()

if user_input == "":
    search_and_copy_v2() 
elif user_input.lower() == "t":
    train_one_vs_all_v2()
else:
    print("Wrong input")

def print_crocodile():
    import os
    import random
    from PIL import Image
    import numpy as np

    # Папка з зображеннями
    folder = "input"  # заміни на свою папку

    # Випадково вибираємо файл
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    random_file = random.choice(files)
    file_path = os.path.join(folder, random_file)

    # Відкриваємо і конвертуємо у відтінки сірого
    img = Image.open(file_path).convert("L")

    # Зменшуємо розмір для кращої ASCII візуалізації
    width, height = img.size
    aspect_ratio = height / width
    new_width = 80
    new_height = int(aspect_ratio * new_width * 0.55)  # 0.55 щоб компенсувати співвідношення символів
    img = img.resize((new_width, new_height))

    # Масив пікселів
    pixels = np.array(img)

    # ASCII символи від темного до світлого
    chars = "@%#*+=-:. "

    # Нормалізуємо пікселі до довжини chars
    def pixel_to_char(pixel):
        return chars[int(pixel / 255 * (len(chars) - 1))]

    # Створюємо ASCII арт рядки
    ascii_art = "\n".join("".join(pixel_to_char(pixel) for pixel in row) for row in pixels)

    print(f"File: {random_file}\n")
    print(ascii_art)

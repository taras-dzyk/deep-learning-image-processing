from model_trainer import train_one_vs_all, train_one_vs_all_v2
from image_searcher import search_and_copy_v2

def print_crocodile():
    import os
    import random

    folder = "_ascii"

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


user_input = input(f'Do you want to train a model of find crocodiles? (type \'t\' for train, hit return for crocodile search. \'s\' for strict )').strip()

if user_input == "":
    search_and_copy_v2() 
elif user_input.lower() == "t":
    train_one_vs_all_v2()
elif user_input.lower() == "s":
    search_and_copy_v2(0.9) 
else:
    print("Wrong input")

def print_crocodile():
    import os
    import random
    from PIL import Image
    import numpy as np

 
    folder = "input"  

   
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    random_file = random.choice(files)
    file_path = os.path.join(folder, random_file)


    img = Image.open(file_path).convert("L")


    width, height = img.size
    aspect_ratio = height / width
    new_width = 80
    new_height = int(aspect_ratio * new_width * 0.55) 
    img = img.resize((new_width, new_height))


    pixels = np.array(img)


    chars = "@%#*+=-:. "


    def pixel_to_char(pixel):
        return chars[int(pixel / 255 * (len(chars) - 1))]

    ascii_art = "\n".join("".join(pixel_to_char(pixel) for pixel in row) for row in pixels)

    print(f"File: {random_file}\n")
    print(ascii_art)

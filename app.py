from model_trainer import train_model, train_one_vs_all, train_one_vs_all_v2
from image_searcher import search, search_and_copy, search_and_copy_v2

print("Welcome to crocodile finder")


user_input = input(f'Do you want to train a model of find crocodiles? (type \'t\' for train, hit return for crocodile search)').strip()

if user_input == "":
    search_and_copy_v2() 
elif user_input.lower() == "t":
    train_one_vs_all_v2()
else:
    print("Wrong input")
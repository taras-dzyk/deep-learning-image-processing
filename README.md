# deep-learning-image-processing
Crocodile finder. 

                    .-._   _ _ _ _ _ _ _ _
         .-''-.__.-'00  '-' ' ' ' ' ' ' ' '-.
         '.___ '    .   .--_'-' '-' '-' _'-' '._
          V: V 'vv-'   '_   '.       .'  _..' '.'.
            '=.____.=_.--'   :_.__.__:_   '.   : :
                    (((____.-'        '-.  /   : :
                                      (((-'\ .' /
                                    _____..'  .'
                                   '-._____.-'


This is console application. 
- scans images from `input` folder 
- identifies crocodiles among them 
- copies crocodile images into `output` folder

Few optins were added: CNN model with custom loss function which prioritizes crockodile class. And model with feature extraction technique. 
Both are using binary classification approach where 1 - crocodile and 0 - all other images

# How to run
You will need poetry and python 3.11

```
poetry env use 3.11.8  
poetry install --no-root
poetry run python main.py
```

# Modes
- Training mode(`t`) Trains and stores model based on `dataset` folder images
- Prediction mode. Scans `input` folder and copies crocodiles to `output` folder

# Project structure
```
deep-learning-image-processing/
├── dataset/ # Тренувальні дані. Їх можна розширювати
│   ├── crocodiles/
│   ├── airplanes/
│   ├── dollar_bills/
│   ├── electric_guitars/
│   └── motorbikes/
├── input/  # Вхідні дані для прогнозування
├── output/ # Вихідні визначені зображення
├── models/
│   ├── weights.h5
│   └── model_binary.keras
├── app.py # Головний файл
├── image_searcher.py
├── model_trainer.py
├── README.md
└── pyproject.toml # Структура проекту і зовнішні модулі

```

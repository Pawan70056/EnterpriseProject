import os
import urllib.request

# Define paths
models_dir = 'C:/Users/ssmp7/Desktop/Enterprise_Project/models'
emotion_model_file = os.path.join(models_dir, 'path_to_emotion_model.h5')
gender_model_file = os.path.join(models_dir, 'path_to_gender_model.h5')
age_model_file = os.path.join(models_dir, 'path_to_age_model.h5')

# Create the models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# URLs to download the model files (Replace these URLs with the actual URLs of your models)
emotion_models_url = 'https://example.com/path_to_emotion_model.h5'
gender_models_url = 'https://example.com/path_to_gender_model.h5'
age_models_url = 'https://example.com/path_to_age_model.h5'

# Download the files if they don't exist
if not os.path.isfile(emotion_model_file):
    print(f"Downloading {emotion_model_file}...")
    urllib.request.urlretrieve(emotion_model_url, emotion_model_file)
    print(f"{emotion_model_file} downloaded.")
else:
    print(f"{emotion_model_file} already exists.")

if not os.path.isfile(gender_model_file):
    print(f"Downloading {gender_model_file}...")
    urllib.request.urlretrieve(gender_model_url, gender_model_file)
    print(f"{gender_model_file} downloaded.")
else:
    print(f"{gender_model_file} already exists.")

if not os.path.isfile(age_model_file):
    print(f"Downloading {age_model_file}...")
    urllib.request.urlretrieve(age_model_url, age_model_file)
    print(f"{age_model_file} downloaded.")
else:
    print(f"{age_model_file} already exists.")

print("Files downloaded and saved to the specified directory.")

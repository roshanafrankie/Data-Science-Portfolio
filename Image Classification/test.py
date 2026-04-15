from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 1. Load the Model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.weights.h5")
print("Loaded model from disk")

# 2. Define the Classification Function
def classify(img_file):
    # Match the training: 128x128 and grayscale
    test_image = image.load_img(img_file, target_size=(128, 128), color_mode='grayscale')
    
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0  # Normalize pixels
    test_image = np.expand_dims(test_image, axis=0) # Add batch dimension
    
    result = model.predict(test_image, verbose=0) # verbose=0 keeps the console clean

    # Sigmoid output logic: 0 to 0.5 = Joker, 0.5 to 1 = Thanos
    # (Based on your index: {'joker': 0, 'thanos': 1})
    if result[0][0] > 0.5:
        prediction = 'Thanos'
    else:
        prediction = 'Joker'
        
    print(f"Prediction: {prediction: <7} | Confidence: {result[0][0]:.4f} | File: {os.path.basename(img_file)}")

# 3. Setup Path and Extensions
path = r'C:\Users\acer\Desktop\Data-Science-Portfolio\Image Classification\Dataset\Lets check' 
valid_extensions = ('.jpg', '.jpeg', '.png')

# 4. Find and Classify Files
print(f"Scanning folder: {path}\n" + "-"*50)

files_found = 0
for r, d, f in os.walk(path):
    for file in f:
        # Check if the file ends with any of our valid extensions
        if file.lower().endswith(valid_extensions):
            full_path = os.path.join(r, file)
            classify(full_path)
            files_found += 1

if files_found == 0:
    print("No matching image files found. Check your file extensions!")
else:
    print("-"*50 + f"\nFinished! Processed {files_found} images.")
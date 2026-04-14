import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# 1. Load the model structure and weights
with open('model.json', 'r') as f:
    loaded_model = model_from_json(f.read())
loaded_model.load_weights("model.weights.h5")

# 2. Define your labels (Must be in the exact order used during training)
labels = ["Apple___Apple_scab", "Apple___Black_rot", ...] 

# 3. Load and prep the image (The "Essential" transformation)
img = image.load_img('im_for_testing_purpose/a.blackrot.JPG', target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0  # Scale pixels to 0-1
img_batch = np.expand_dims(img_array, axis=0)

# 4. Predict
predictions = loaded_model.predict(img_batch)
result_index = np.argmax(predictions)
print(f"Prediction: {labels[result_index]} (Confidence: {np.max(predictions)*100:.2f}%)")
from numpy import loadtxt
from keras.models import model_from_json
import joblib
import numpy as np

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("model.weights.h5")
print("Loaded model from disk")

sc = joblib.load('scaler.bin')
x_scaled = sc.transform(x) 

predictions = model.predict(x_scaled)

for i in range(20, 25):
    
    predicted_class = int(np.round(predictions[i][0]))
    print('%s => %d (expected %d)' % (x[i].tolist(), predicted_class, y[i]))
    
new_patient = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
new_patient_scaled = sc.transform(new_patient) 

prediction = model.predict(new_patient_scaled)
print("Diabetic Probability:", int(np.round(predictions[0][0])))
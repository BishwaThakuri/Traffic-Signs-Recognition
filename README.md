# Traffic Sign Recognition using CNN & OpenCV

## 📌 Project Overview

This project implements a real-time traffic sign recognition system using Convolutional Neural Networks (CNNs) trained on a dataset of over 35,000 images spanning 43 different traffic sign classes. The trained model is then used for real-time traffic sign detection using a webcam with OpenCV.

## 🚀 Features

- Dataset of 43 Traffic Sign Classes
- CNN Model Training with TensorFlow & Keras
- Real-Time Traffic Sign Recognition using OpenCV
- Live Camera Feed for Prediction

## 💂️ Dataset

The dataset contains:

- `data/` - Folder with images of traffic signs categorized into different classes.
- `labels.csv` - File containing class labels and names for traffic signs.

### Dataset Preprocessing:

- Image resizing to 32x32 pixels
- Normalization of pixel values
- One-hot encoding of labels

## 🛠️ Installation & Setup

### 1⃣ Install Dependencies

```bash
pip install tensorflow opencv-python numpy pandas matplotlib
```

### 2⃣ Clone the Repository

```bash
git clone https://github.com/BishwaThakuri/Traffic-Signs-Recognition.git
cd traffic-sign-recognition
```

### 3⃣ Train the Model

```bash
jupyter notebook Real Time Traffic Sign Recognition- Train code.ipynb
```
This will open the Jupyter Notebook and allow you to train the CNN model on the dataset, saving it as model_trained.p.

### 4⃣ Test the Model (Real-Time Prediction)

```bash
jupyter notebook Real Time Traffic Sign Recognition - Test Code.ipynb
```
This will open the Jupyter Notebook for real-time traffic sign recognition using a webcam.


## 🧠 Model Architecture

```bash
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 1)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```
- `Loss Function:` Categorical Crossentropy
- `Optimizer:` Adam
- `Metrics:` Accuracy

## 🎥 Real-Time Traffic Sign Detection
Using OpenCV, the trained model can recognize traffic signs in real-time through a webcam.

```bash
frameWidth, frameHeight = 640, 480
brightness = 150
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

with open("model_trained.p", "rb") as pickle_in:
    model = pickle.load(pickle_in)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

def getClassName(classNo):
    class_labels = {  
        0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h',
        3: 'Speed Limit 60 km/h', 4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h',
        6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h', 8: 'Speed Limit 120 km/h',
        9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons', 
        11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield', 
        14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited', 
        17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left', 
        20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road', 
        23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work', 
        26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 29: 'Bicycles crossing',
        30: 'Beware of ice/snow', 31: 'Wild animals crossing', 32: 'End of all speed and passing limits',
        33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
        37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
        41: 'End of no passing', 42: 'End of no passing by vehicles over 3.5 metric tons'
    }
    return class_labels.get(classNo, "Unknown")

while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Error: Could not read from camera.")
        break
    try:
        img = cv2.resize(imgOriginal, (32, 32))
        img = preprocess_image(img)
        img = img.reshape(1, 32, 32, 1)
        
        predictions = model.predict(img)
        classIndex = np.argmax(predictions[0])
        probabilityValue = np.max(predictions[0])
        
        if probabilityValue > threshold:
            className = getClassName(classIndex)
            probabilityText = f"{round(probabilityValue * 100, 2)}%"
            
            cv2.putText(imgOriginal, f"CLASS: {className}", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOriginal, f"PROBABILITY: {probabilityText}", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Result", imgOriginal)
    except Exception as e:
        print(f"Error during processing: {e}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📊 Performance & Evaluation
- `Test Loss:` 0.0321
- `Test Accuracy:` 98.94%
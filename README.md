# 🚦 Traffic Sign Recognition using MobileNetV2  

This project implements a **Traffic Sign Recognition System** using **Deep Learning (CNN & MobileNetV2)**. The model can classify different traffic signs (e.g., Stop, Speed Limit, Yield, etc.) from images. It is built and trained in **Google Colab** using **TensorFlow/Keras**.  

---

## 📌 Features  
- ✅ Train traffic sign recognition models (Custom CNN + MobileNetV2).  
- ✅ Transfer Learning with **MobileNetV2** for high accuracy.  
- ✅ Preprocessing with OpenCV (`cv2`) and MobileNetV2’s `preprocess_input`.  
- ✅ Model saving in `.h5` format for reusability.  
- ✅ Image upload and **real-time prediction** in Google Colab.  

---

## 📂 Project Structure  

```
├── best_custom_cnn.h5               # Best trained Custom CNN model
├── custom_cnn_final.h5              # Final Custom CNN model
├── best_mobilenetv2.h5              # Best trained MobileNetV2 model
├── best_mobilenetv2_finetuned.h5    # Best fine-tuned MobileNetV2
├── mobilenetv2_final.h5             # Final MobileNetV2 model
├── dataset/                          # Training & testing dataset
│   ├── Train/                        # Training images
│   ├── Test/                         # Testing images
│   └── SignNames.csv                 # Mapping of class IDs to sign names
└── traffic_sign_prediction.ipynb    # Colab notebook with training & prediction
```

---

## ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition
```

Install dependencies:  
```bash
pip install tensorflow opencv-python matplotlib pandas
```

---

## 🧑‍💻 Training

To train the model (in Colab or local):  

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Example: Load MobileNetV2 base
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128,128,3))

# Add custom layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=20)
```

---

## 🔍 Prediction

Upload an image in Google Colab and run prediction:  

```python
from google.colab import files
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
import matplotlib.pyplot as plt

uploaded = files.upload()

for fn in uploaded.keys():
    img_path = fn
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))  # match training size

    # Preprocess
    img_input = np.expand_dims(img_resized, axis=0)
    img_input = preprocess_input(img_input)

    # Prediction
    pred = model.predict(img_input)
    pred_class = np.argmax(pred)

    # Display
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Predicted: {id_to_sign[pred_class]}")
    plt.show()

    print(f"Predicted: Class {pred_class} → {id_to_sign[pred_class]}")
```

---

## 📊 Results

- **MobileNetV2 (Fine-tuned):** Best accuracy  
- **Custom CNN:** Lightweight but slightly lower performance  

| Model                  | Accuracy  |
|------------------------|-----------|
| Custom CNN             | ~92%      |
| MobileNetV2            | ~96%      |
| MobileNetV2 (Fine-tuned)| ~98%     |

---

## 📈 Example Predictions

| Input Image          | Predicted Sign      |
|---------------------|------------------|
| 🛑 Stop Sign         | Stop             |
| 🚫 Speed Limit 50    | Speed Limit 50   |
| ⚠️ Yield Sign        | Yield            |


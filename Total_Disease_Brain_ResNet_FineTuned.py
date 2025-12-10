import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam

# -------------------- Parameters -------------------- #
IMAGE_SIZE = 150
DATASET_PATH = "dataset"
MODEL_PATH = "model/Brain_Disease_Classification_Using_VGG16_Model.h5"
CLASS_NAMES = ['Alzheimer','Brain_Tumor', 'Healthy','MS']

# -------------------- Load Dataset -------------------- #
X = []
y = []

for label in CLASS_NAMES:
    folder = os.path.join(DATASET_PATH, label)
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        if len(img.shape)==2 or img.shape[2]==1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img / 255.0
        X.append(img)
        y.append(CLASS_NAMES.index(label))

X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y, random_state=42)
y = to_categorical(y, num_classes=len(CLASS_NAMES))

# -------------------- Train/Test Split -------------------- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Build Model -------------------- #
vgg = VGG16(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(len(CLASS_NAMES), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------- Train -------------------- #
hist = model.fit(X_train, y_train, validation_split=0.1, epochs=15, batch_size=16)

# -------------------- Save Model -------------------- #
if not os.path.exists("model"):
    os.makedirs("model")
model.save(MODEL_PATH)
print("Model saved at", MODEL_PATH)

# -------------------- Evaluate -------------------- #
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

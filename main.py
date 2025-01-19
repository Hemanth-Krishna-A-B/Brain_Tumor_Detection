import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import itertools


data_dir = './data/cropped'
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 200

def load_images_from_directory(data_dir, categories, image_size):
    images = []
    labels = []
    for label in categories:
        category_path = os.path.join(data_dir, label)
        for file in os.listdir(category_path):
            image_path = os.path.join(category_path, file)
            image = cv2.imread(image_path, 0) 
            if image is not None:
                image = cv2.bilateralFilter(image, 2, 50, 50)  
                image = cv2.applyColorMap(image, cv2.COLORMAP_BONE) 
                image = cv2.resize(image, (image_size, image_size))
                images.append(image)
                labels.append(categories.index(label))
    return np.array(images), np.array(labels)

train_data_dir = os.path.join(data_dir, 'Training')
test_data_dir = os.path.join(data_dir, 'Testing')

x_train, y_train = load_images_from_directory(train_data_dir, categories, image_size)
x_test, y_test = load_images_from_directory(test_data_dir, categories, image_size)


x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(categories))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(categories))

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)


datagen.fit(x_train)


IMG_SIZE = (200, 200)
conv_base = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights='imagenet'
)


for layer in conv_base.layers:
    layer.trainable = True


model = tf.keras.Sequential([
    conv_base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(categories), activation="softmax")
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


callbacks = [
    tf.keras.callbacks.ModelCheckpoint('model_weights.keras', monitor='val_loss', mode='min', verbose=1, save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, mode='min', min_lr=1e-10)
]

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=8),
    validation_data=(x_val, y_val),
    epochs=20,
    callbacks=callbacks
)


plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.show()

plt.figure(figsize=[8, 6])
plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()


loss, acc = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {acc}')


predicted_classes = np.argmax(model.predict(x_test), axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(y_true_classes, predicted_classes, target_names=categories))

conf_matrix = confusion_matrix(y_true_classes, predicted_classes)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(conf_matrix, categories)
plt.show()
model.save('brain_tumor_classifier.h5')

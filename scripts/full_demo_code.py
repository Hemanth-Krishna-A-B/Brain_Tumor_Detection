import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras as krs
from sklearn.metrics import classification_report, confusion_matrix
from keras import EarlyStopping

# get image path and labels from a diretory
def get_data_labels(base_dir):
    image_paths = []
    class_labels = []

    for label in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, img_file))
                class_labels.append(label)

    return pd.DataFrame({'image': image_paths, 'class': class_labels})

# class distribtion
def plot_class_distribution(data):
    sns.countplot(x='class', data=data)
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.show()

# craete and compile the model
def create_model(input_shape, num_classes):
    model = krs.Sequential([
        krs.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        krs.MaxPooling2D(pool_size=(2, 2)),
        krs.Dropout(0.25),

        krs.Conv2D(64, (3, 3), activation='relu'),
        krs.MaxPooling2D(pool_size=(2, 2)),
        krs.Dropout(0.25),

        krs.Conv2D(128, (3, 3), activation='relu'),
        krs.MaxPooling2D(pool_size=(2, 2)),
        krs.Dropout(0.4),

        krs.Flatten(),
        krs.Dense(128, activation='relu'),
        krs.Dropout(0.5),
        krs.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=krs.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

# Confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Main execution example
if __name__ == "__main__":
    base_dir = "./dataset" 

    # Load data
    data = get_data_labels(base_dir)
    plot_class_distribution(data)

    # Data preprocessing
    train_datagen = krs.ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_dataframe(
        data,
        x_col='image',
        y_col='class',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_dataframe(
        data,
        x_col='image',
        y_col='class',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # train the model
    model = create_model(input_shape=(150, 150, 3), num_classes=len(train_generator.class_indices))

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
    )

    # Ploting  trained history
    plot_training_history(history)

    # Evalulating the model
    val_labels = validation_generator.classes
    val_preds = np.argmax(model.predict(validation_generator), axis=1)
    plot_confusion_matrix(val_labels, val_preds, class_names=list(train_generator.class_indices.keys()))

    # Classification report
    print(classification_report(val_labels, val_preds, target_names=list(train_generator.class_indices.keys())))

from sklearn.metrics import classification_report
from preprocess import get_data_labels
from train import create_model
from visualize import plot_class_distribution, plot_confusion_matrix, plot_training_history
import numpy as np
from keras import ImageDataGenerator
from keras import EarlyStopping


if __name__ == "__main__":
    base_dir = "./dataset"  

    
    data = get_data_labels(base_dir)
    plot_class_distribution(data)

    
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

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

    
    model = create_model(input_shape=(150, 150, 3), num_classes=len(train_generator.class_indices))

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
    )

    
    plot_training_history(history)

    val_labels = validation_generator.classes
    val_preds = np.argmax(model.predict(validation_generator), axis=1)
    plot_confusion_matrix(val_labels, val_preds, class_names=list(train_generator.class_indices.keys()))

    print(classification_report(val_labels, val_preds, target_names=list(train_generator.class_indices.keys())))
import os
import numpy as np
import pandas as pd
import tensorflow as tf


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

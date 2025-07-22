import os
import numpy as np
from PIL import Image

# label mapping for 4 races
race_to_index = {
    "Azael": 0,
    "Morvid": 1,
    "Navaran": 2,
    "Scroom": 3
}

def load_dataset(data_dir, image_size=(64, 64)):
    X = []  # image data
    y = []  # labels

    for race_name, label in race_to_index.items():
        race_dir = os.path.join(data_dir, race_name)

        if not os.path.isdir(race_dir):
            print(f"Warning: Folder '{race_dir}' not found, skipping.")
            continue

        for filename in os.listdir(race_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(race_dir, filename)

                # resize and rgb conversion
                img = Image.open(path).convert('RGB')
                img = img.resize(image_size)

                # convert to np array and normalize form [0,1]
                img_array = np.asarray(img) / 255.0
                X.append(img_array)
                y.append(label)

    return np.array(X), np.array(y)

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

if __name__ == "__main__":
    X, y = load_dataset("james_dataset") 
    y_onehot = one_hot_encode(y, num_classes=4)

    np.savez("races_dataset.npz", images=X, labels=y_onehot)
    print(f"Saved {X.shape[0]} images to 'races_dataset.npz'")

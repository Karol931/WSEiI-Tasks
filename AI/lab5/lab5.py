import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import KFold
import keras as k
import tensorflow as tf


def create_augmentation(rotation_range, width_shift_range, height_shift_range, zoom_range):
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=rotation_range,             # Zakres obrotów
        width_shift_range=width_shift_range,       # Przesunięcie w poziomie
        height_shift_range=height_shift_range,     # Przesunięcie w pionie
        zoom_range=zoom_range                      # Skalowanie
    )


def cnn_model(num_filters):
    drop_dense=0.5
    drop_conv=0
    
    model = k.models.Sequential()
    model.add(k.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_regularizer=None,
    input_shape=(32, 32, 3),padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.Conv2D(num_filters, (3, 3),
    activation='relu',kernel_regularizer=None,padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(drop_conv))
    model.add(k.layers.Conv2D(2*num_filters, (3, 3),
    activation='relu',kernel_regularizer=None,padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.Conv2D(2*num_filters, (3, 3),
    activation='relu',kernel_regularizer=None,padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(drop_conv))
    model.add(k.layers.Conv2D(4*num_filters, (3, 3),
    activation='relu',kernel_regularizer=None,padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.Conv2D(4*num_filters, (3, 3),
    activation='relu',kernel_regularizer=None,padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(drop_conv))
    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(512, activation='relu',kernel_regularizer=None))
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Dropout(drop_dense))
    model.add(k.layers.Dense(3, activation='softmax'))
    model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=k.optimizers.Adam(learning_rate=0.001,decay=0, beta_1=0.9, beta_2=0.999,
    epsilon=1e-08)
    )
    
    return model


(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

selected_classes = [0, 1, 2]
train_mask = np.isin(y_train_full.flatten(), selected_classes)
test_mask = np.isin(y_test_full.flatten(), selected_classes)

x_train = x_train_full[train_mask][:2000]
y_train = y_train_full[train_mask][:2000]
x_test = x_test_full[test_mask][:500]
y_test = y_test_full[test_mask][:500]

mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

model = cnn_model(32)

model.summary()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

batch_size = 32
epochs = 20

print("X_train_split shape:", x_train.shape)
print("X_val shape:", x_val.shape)
print("y_train_split shape:", y_train.shape)
print("y_val shape:", y_val.shape)

datagen = create_augmentation(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

datagen.fit(x_train)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_val, y_val),
    epochs=epochs,
    steps_per_epoch=len(x_train) // batch_size,
    verbose=1
)

rotation_range_list = [10, 20, 30]
width_shift_range_list = [0.1, 0.2, 0.3]
height_shift_range_list = [0.1, 0.2, 0.3]
zoom_range_list = [0.1, 0.2, 0.3]

random_trials = 5
kf = KFold(n_splits=3, shuffle=True, random_state=42)

best_accuracy = 0
best_params = {}

for _ in range(random_trials):
    rotation_range = random.choice(rotation_range_list)
    width_shift_range = random.choice(width_shift_range_list)
    height_shift_range = random.choice(height_shift_range_list)
    zoom_range = random.choice(zoom_range_list)

    datagen = create_augmentation(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range
    )

    accuracies = []

    for train_idx, val_idx in kf.split(x_train, y_train):
        X_train_fold, X_val_fold = x_train[train_idx], x_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        datagen.fit(X_train_fold)

        model = cnn_model(3)
        model.fit(
            datagen.flow(X_train_fold, y_train_fold, batch_size=batch_size),
            validation_data=(X_val_fold, y_val_fold),
            epochs=5,  
            steps_per_epoch=len(X_train_fold) // batch_size,
            verbose=0
        )

        val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        accuracies.append(val_accuracy)

    mean_accuracy = np.mean(accuracies)

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_params = {
            'rotation_range': rotation_range,
            'width_shift_range': width_shift_range,
            'height_shift_range': height_shift_range,
            'zoom_range': zoom_range
        }

print("Najlepsze parametry:", best_params)
print("Najlepsza dokładność:", best_accuracy)


optimal_datagen = create_augmentation(
    rotation_range=best_params['rotation_range'],
    width_shift_range=best_params['width_shift_range'],
    height_shift_range=best_params['height_shift_range'],
    zoom_range=best_params['zoom_range']
)

optimal_datagen.fit(x_train)

final_model = cnn_model(3)
final_history = final_model.fit(
    optimal_datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_test, y_test),
    epochs=epochs,
    steps_per_epoch=len(x_train) // batch_size,
    verbose=1
)

test_loss, test_accuracy = final_model.evaluate(x_test, y_test, verbose=1)
print("Końcowa dokładność na zbiorze testowym:", test_accuracy)
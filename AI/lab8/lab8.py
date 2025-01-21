import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_shape = (64, 64, 3)
latent_dim = 100
batch_size = 64
epochs = 10000
sample_interval = 1000

def load_data(data_dir, img_shape):
    images = []
    for filename in os.listdir(data_dir):
        img = load_img(os.path.join(data_dir, filename), target_size=img_shape[:2])
        img_array = img_to_array(img) / 127.5 - 1.0  
        images.append(img_array)
    return np.array(images)

data_dir = "./cats"  
x_train = load_data(data_dir, img_shape)
print(f"Shape of training data: {x_train.shape}")

def build_generator(latent_dim):
    model = Sequential([
        Dense(8 * 8 * 256, activation="relu", input_dim=latent_dim),
        Reshape((8, 8, 256)),
        Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"),
        Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),
        Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", activation="relu"),
        Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation="tanh")
    ])
    return model

def build_discriminator(img_shape):
    model = Sequential([
        Conv2D(32, kernel_size=4, strides=2, padding="same", input_shape=img_shape),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.25),
        Conv2D(64, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.25),
        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.25),
        Flatten(),
        Dense(1, activation="sigmoid")
    ])
    return model

generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"])

discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy")

def generate_and_save_images(generator, latent_dim, epoch, save_dir="generated_images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    noise = np.random.normal(0, 1, (25, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  
    
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(generated_images[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch}.png")
    plt.close()

real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise)
    
    d_loss_real = discriminator.train_on_batch(real_images, real)
    d_loss_fake = discriminator.train_on_batch(generated_images, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real)
    
    if epoch % 100 == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")
    
    if epoch % sample_interval == 0:
        generate_and_save_images(generator, latent_dim, epoch)

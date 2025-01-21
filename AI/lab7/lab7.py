import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0  
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  
x_test = np.expand_dims(x_test, -1)

latent_dim = 2

def build_encoder(latent_dim):
    encoder_input = Input(shape=(28, 28, 1))
    x = Flatten()(encoder_input)
    x = Dense(200, activation="relu")(x)
    x = Dense(200, activation="relu")(x)
    latent_output = Dense(latent_dim, activation="tanh")(x)
    return Model(encoder_input, latent_output, name="Encoder")

def build_decoder(latent_dim):
    decoder_input = Input(shape=(latent_dim,))
    x = Dense(200, activation="relu")(decoder_input)
    x = Dense(200, activation="relu")(x)
    x = Dense(28 * 28, activation="relu")(x)
    decoder_output = Reshape((28, 28, 1))(x)
    return Model(decoder_input, decoder_output, name="Decoder")

encoder = build_encoder(latent_dim)
decoder = build_decoder(latent_dim)
autoencoder_input = Input(shape=(28, 28, 1))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)

autoencoder = Model(autoencoder_input, decoded, name="Autoencoder")
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

history = autoencoder.fit(
    x_train, x_train,
    epochs=30,
    batch_size=256,
    validation_data=(x_test, x_test),
    verbose=1
)

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss and Validation Loss")
plt.show()

reconstructed = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")
    
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].squeeze(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")
plt.show()

latent_space = encoder.predict(x_test)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_space[:, 0], latent_space[:, 1], c=y_test, cmap="viridis", s=2)
plt.colorbar(scatter, label="Class Label")
plt.title("Latent Space Visualization")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()

random_latent_vectors = np.random.uniform(-1, 1, size=(10, latent_dim))
random_decoded_images = decoder.predict(random_latent_vectors)

plt.figure(figsize=(20, 4))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(random_decoded_images[i].squeeze(), cmap="gray")
    plt.title("Random")
    plt.axis("off")
plt.show()

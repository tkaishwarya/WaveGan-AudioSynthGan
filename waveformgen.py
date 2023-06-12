import tensorflow as tf
import numpy as np
import deeplake

# Load the NSynth datasets
train_ds = deeplake.load("hub://activeloop/nsynth-train")
test_ds = deeplake.load("hub://activeloop/nsynth-test")
val_ds = deeplake.load("hub://activeloop/nsynth-val")

# Extract a limited number of samples
num_samples = 100

train_ds_subset = train_ds[:num_samples]
test_ds_subset = test_ds[:num_samples]
val_ds_subset = val_ds[:num_samples]

# Define the functions to extract audio and metadata
def extract_audio(ds):
    audio = []
    for item in ds:
        print("Still running audio")
        audio.append(item["audios"].numpy())
    return np.array(audio)

def extract_metadata(ds):
    metadata = []
    for item in ds:
        print("Still running metadata")
        metadata.append({
            'instrument': item['instrument'],
            'instrument_family': item['instrument_family'],
            'instrument_source': item['instrument_source'],
            'note': item['note'],
            'pitch': item['pitch'],
            'qualities': item['qualities'],
            'sample_rate': item['sample_rate'],
            'velocity': item['velocity']
        })
    return metadata

# Extract audio and metadata from the subsets of the datasets
train_audio = extract_audio(train_ds_subset)
train_metadata = extract_metadata(train_ds_subset)

test_audio = extract_audio(test_ds_subset)
test_metadata = extract_metadata(test_ds_subset)

val_audio = extract_audio(val_ds_subset)
val_metadata = extract_metadata(val_ds_subset)

# Define the generator model
def build_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(100,)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape((16, 16, 1)))
    model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', activation='tanh'))
    return model

# Define the discriminator model
def build_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=(64, 64, 1)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model

# Define the loss function
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(real_samples, fake_samples, discriminator):
    alpha = tf.random.uniform([real_samples.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
    interpolated_samples = alpha * real_samples + (1 - alpha) * fake_samples

    with tf.GradientTape() as tape:
        tape.watch(interpolated_samples)
        interpolated_output = discriminator(interpolated_samples, training=True)

    gradients = tape.gradient(interpolated_output, interpolated_samples)
    squared_gradients = tf.square(gradients)
    sum_squared_gradients = tf.reduce_sum(squared_gradients, axis=[1, 2, 3])
    norm = tf.sqrt(sum_squared_gradients)
    penalty = tf.reduce_mean(tf.square(norm - 1))
    return penalty


# Define the generator and discriminator optimizers
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training loop
def train_wavegan(generator, discriminator, audio_data, num_epochs=10, batch_size=32):
    num_steps = len(audio_data) // batch_size

    for epoch in range(num_epochs):
        for step in range(num_steps):
            # Extract a batch of audio data
            audio_batch = audio_data[step * batch_size: (step + 1) * batch_size]
            audio_batch = tf.expand_dims(audio_batch, axis=-1)
            audio_batch = tf.image.resize(audio_batch, (64, 64))  # Resize the audio batch

            # Generate random noise as input to the generator
            noise = tf.random.normal([batch_size, 100])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_audio = generator(noise, training=True)
                generated_audio = tf.image.resize(generated_audio, (64, 64))  # Resize the generated audio

                real_output = discriminator(audio_batch, training=True)
                fake_output = discriminator(generated_audio, training=True)

                gen_loss = -tf.reduce_mean(fake_output)
                disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                disc_loss += 10 * gradient_penalty(audio_batch, generated_audio, discriminator)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            if step % 100 == 0:
                print(f'Epoch {epoch+1}, Step {step}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')

# Create the generator and discriminator models
generator = build_generator_model()
discriminator = build_discriminator_model()

# Define other necessary variables
num_epochs = 10
batch_size = 32

# Train the WaveGAN model
train_wavegan(generator, discriminator, train_audio, num_epochs, batch_size)

def generate_audio(generator, num_samples):
    noise = tf.random.normal([num_samples, 100])
    generated_audio = generator(noise, training=False)
    generated_audio = tf.squeeze(generated_audio, axis=-1).numpy()
    return generated_audio

def gradient_penalty(real_samples, fake_samples, discriminator):
    alpha = tf.random.uniform([real_samples.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
    interpolated_samples = alpha * real_samples + (1 - alpha) * fake_samples

    with tf.GradientTape() as tape:
        tape.watch(interpolated_samples)
        interpolated_output = discriminator(interpolated_samples, training=True)

    gradients = tape.gradient(interpolated_output, interpolated_samples)
    squared_gradients = tf.square(gradients)
    sum_squared_gradients = tf.reduce_sum(squared_gradients, axis=[1, 2, 3])
    norm = tf.sqrt(sum_squared_gradients)
    penalty = tf.reduce_mean(tf.square(norm - 1))
    return penalty

import os
import numpy as np
from scipy.io import wavfile

def save_audio_samples(audio_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    for i, audio in enumerate(audio_samples):
        filename = f"generated_sample_{i+1}.wav"
        filepath = os.path.join(output_dir, filename)
        scaled_audio = np.int16(audio / np.max(np.abs(audio)) * 32767)  # Scale audio to 16-bit range
        wavfile.write(filepath, 16000, scaled_audio)

# Generate audio samples
num_samples = 5
generated_audio = generate_audio(generator, num_samples)

# Save the generated audio samples as WAV files
output_dir = "./generated_audio/"
save_audio_samples(generated_audio, output_dir)


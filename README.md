# WaveGan-AudioSynthGan

This project implements the WaveGAN model for audio generation using TensorFlow. It includes the necessary components such as the generator and discriminator models, the training loop, the loss function, and the gradient penalty calculation.

1. Data Loading and Preprocessing:
   - The code uses the `deeplake` library to load the NSynth dataset. NSynth is a large-scale dataset of musical notes created by Google. It consists of audio waveforms and corresponding metadata for various musical instruments.
   - The dataset is loaded using the `deeplake.load` function, which takes a URL or a local path as input. In this case, it loads the NSynth training, testing, and validation datasets.
   - A subset of the datasets is extracted by slicing the original datasets with the `[:num_samples]` notation. This limits the number of samples to a specified number (`num_samples` variable).

2. Data Extraction:
   - Two functions are defined to extract audio and metadata from the subsets of the datasets: `extract_audio` and `extract_metadata`.
   - The `extract_audio` function iterates over the dataset and appends the audio waveforms to a list.
   - The `extract_metadata` function iterates over the dataset and appends relevant metadata to a list as a dictionary.
   - The extracted audio and metadata are stored in separate variables for the training, testing, and validation subsets.

3. Generator Model:
   - The `build_generator_model` function defines the generator model using the `tf.keras.Sequential` API.
   - The generator is a sequential model with a series of layers.
   - It starts with a fully connected dense layer with 256 units, followed by a leaky ReLU activation.
   - The output of the dense layer is reshaped into a 16x16x1 feature map.
   - Two transposed convolutional layers are added with 128 and 64 filters, respectively, using a 4x4 kernel, stride of 2, and "same" padding.
   - Each transposed convolutional layer is followed by a leaky ReLU activation.
   - The final layer is a transposed convolutional layer with 1 filter, a 4x4 kernel, stride of 2, "same" padding, and a hyperbolic tangent activation function.
   - The generator model is returned by the function.

4. Discriminator Model:
   - The `build_discriminator_model` function defines the discriminator model using the `tf.keras.Sequential` API.
   - The discriminator is a sequential model with a series of convolutional layers.
   - It starts with a 2D convolutional layer with 64 filters, a 4x4 kernel, stride of 2, "same" padding, and an input shape of 64x64x1.
   - The convolutional layer is followed by a leaky ReLU activation.
   - Two more convolutional layers are added with 128 and 256 filters, respectively, using a 4x4 kernel, stride of 2, and "same" padding.
   - Each convolutional layer is followed by a leaky ReLU activation.
   - The feature maps are flattened and passed through a fully connected dense layer with 1 unit.
   - The discriminator model is returned by the function.

5. Loss Function:
   - The `wasserstein_loss` function defines the Wasserstein loss, which is used as the loss function for both the generator and discriminator models.
   - The function takes in the true labels (`y_true`) and predicted labels (`y_pred`).
   - The loss is computed as the mean of the element-wise multiplication of `y_true` and `y_pred`.

6. Gradient Penalty:
   - The `gradient_penalty` function calculates the gradient penalty, which is used to enforce the Lipschitz constraint on the discriminator.
   - The function

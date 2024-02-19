# From: https://www.tensorflow.org/tutorials/keras/save_and_load
## Setup
### Installs and imports
# ! pip install pyyaml h5py # Required to save models in HDF5 format

import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

### Get an ex. dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1_000]
test_labels = test_labels[:1_000]

train_images = train_images[:1_000].reshape(-1, 28 * 28) / 255.
test_images = test_images[:1_000].reshape(-1, 28 * 28) / 255.

### Define a model
# Define a simple sequential model
def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

## Save checkpoints during training
### Checkpoint callback usage
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model w/ the new callback
model.fit(train_images, train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback]) # Pass callback to training
    # This may generate warnings related to saving the state of the optimizer.
    # (...) are in place to discourage outdated usage, and can be ignored.

os.listdir(checkpoint_dir)

# As long as 2 models share the same architecture, you can share the weights bw. them. So, when restoring a model
# from weights-only, 1) create a model w/ the same architecture as the original model, and
# 2) set its weights.

# 1) Now rebuild a fresh, untrained model and evaluate it on the test set. (...) will perform at chance levels (~10% acc):

# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels,
                           verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# 2) Then load the weights from the checkpoint and re-evaluate:

# Load the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels,
                           verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

### Checkpoint callback options
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Calculate the number of batches per epoch
import math
n_batches = len(train_images) / batch_size
n_batches = math.ceil(n_batches) # round up the number of batches to the nearest whole integer

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq = 5 * n_batches,
                                                 verbose=1)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model w/ the new callback
model.fit(train_images, train_labels,
          epochs=50,
          batch_size=batch_size,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback],
          verbose=0)

os.listdir(checkpoint_dir)

latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels,
                           verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

## What are these files?

## Manually save weights
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels,
                           verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

## Save the entire model
### New high-level .keras format
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels,
          epochs=5)

# Save the entire model as a `.keras` zip archive.
model.save('my_model.keras')

# Reload a fresh Keras model from the `.keras` zip archive:
new_model = tf.keras.models.load_model('my_model.keras')

# Show the model architecture
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels,
                               verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)

### SavedModel format
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels,
          epochs=5)

# Save the entire model as a SavedModel.
# ! mkdir -p saved_model
model.save('saved_model/my_model')

# my_model directory
# ls saved_model

# Contains an assets folder, saved_model.pb, and variables folder.
# ls saved_model/my_model

# Reload a fresh Keras model from the saved model
new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels,
                               verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)

### HDF5 format
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels,
          epochs=5)

# Save the entire model to a HDF5 file.
# The `.h5` extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')

# Recreate the exact same model, incl. its weights, and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

# Check its accuracy
loss, acc = new_model.evaluate(test_images, test_labels,
                               verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))




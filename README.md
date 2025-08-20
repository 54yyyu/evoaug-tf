# EvoAug-TF
Evolution-inspired data augmentations for TensorFlow-based models for regulatory genomics ([paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02941-w)). This library provides a dataloader-based approach for applying augmentations during training, making it easy to integrate with any TensorFlow/Keras model and training pipeline.

#### Install:

```
pip install evoaug-tf
```


#### Dependencies:

```
tensorflow >= 2.11.0
numpy >= 1.21.0
```

#### Example Usage

```python
import os
from evoaug_tf import evoaug, augment
import tensorflow as tf
from tensorflow import keras

# Define your augmentations
augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=30),
    augment.RandomRC(rc_prob=0.5),
    augment.RandomInsertion(insert_min=0, insert_max=20),  # Maintains sequence length by trimming
    augment.RandomTranslocation(shift_min=0, shift_max=20),
    augment.RandomNoise(noise_mean=0, noise_std=0.3),
    augment.RandomMutation(mutate_frac=0.05)
]

# Create augmented datasets
train_dataset = evoaug.EvoAugDataset.create_dataset(
    x_train, y_train, 
    augment_list=augment_list, 
    batch_size=32,
    max_augs_per_seq=1,
    hard_aug=True,
    apply_augmentations=True,  # Enable augmentations for training
    shuffle=True  # Shuffle training data
)

val_dataset = evoaug.EvoAugDataset.create_dataset(
    x_valid, y_valid,
    augment_list=None,  # No augmentations for validation
    batch_size=32,
    apply_augmentations=False,  # Disable augmentations for validation
    shuffle=False  # Don't shuffle validation data
)

# Create your Keras model (standard approach)
def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Conv1D(64, 15, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling1D(5),
        keras.layers.Conv1D(128, 10, activation='relu'),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    return model

# Calculate input shape (insertions now maintain original sequence length)
input_shape = (x_train.shape[1], x_train.shape[2])
model = create_model(input_shape)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Set up callbacks
es_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5, 
    min_lr=1e-7,
    mode='min',
    verbose=1
)

# Train model with augmentations
history = model.fit(
    train_dataset.dataset,
    epochs=100,
    validation_data=val_dataset.dataset,
    callbacks=[es_callback, reduce_lr]
)

# Save model weights
model.save_weights(os.path.join(output_dir, exp_name + "_aug.h5"))

# Fine-tuning: Create dataset without augmentations
finetune_train_dataset = evoaug.EvoAugDataset.create_dataset(
    x_train, y_train,
    augment_list=None,  # No augmentations for fine-tuning
    batch_size=32,
    apply_augmentations=False,
    shuffle=True
)

# Update optimizer for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='mse',
    metrics=['mae']
)

# Fine-tune model
history_ft = model.fit(
    finetune_train_dataset.dataset,
    epochs=20,
    validation_data=val_dataset.dataset,
    callbacks=[es_callback]
)

# Save fine-tuned weights
model.save_weights(os.path.join(output_dir, exp_name + "_finetune.h5"))
```

#### Advanced Usage

```python
# Custom dataset configuration
custom_dataset = evoaug.EvoAugDataset(
    x_train, y_train,
    augment_list=augment_list,
    batch_size=64,
    max_augs_per_seq=2,        # Apply up to 2 augmentations per sequence
    hard_aug=False,            # Random number of augmentations (1 to max_augs_per_seq)
    shuffle=True,              # Shuffle the dataset
    buffer_size=10000,         # Shuffle buffer size
    apply_augmentations=True   # Enable augmentations
)

# Access the underlying tf.data.Dataset for advanced operations
tf_dataset = custom_dataset.dataset
tf_dataset = tf_dataset.cache()  # Add caching
tf_dataset = tf_dataset.repeat()  # Repeat indefinitely

# Use with model.fit()
model.fit(tf_dataset, steps_per_epoch=100, epochs=50)
```

#### Key Benefits of the Dataloader Approach

- **Framework Agnostic**: Works with any TensorFlow/Keras model and training loop
- **Better Integration**: Easy to integrate into existing ML pipelines
- **Performance Optimized**: Leverages tf.data.Dataset optimizations (prefetch, parallel map, etc.)
- **Flexible**: Separate augmentation logic from model definition
- **Clean Separation**: Clear distinction between data preprocessing and model training

#### Example on Google Colab:

- Example DeepSTARR analysis: https://colab.research.google.com/drive/11TA02v-azuqAIV5s3sCbWTH-C5W7_KMA 
- Example ChIP-seq analysis: https://colab.research.google.com/drive/1fzpH2Qv8RFNzMvIDRJUGnUTaacFMIJBV

#### Original EvoAug in PyTorch:
- Paper: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02941-w 
- Example DeepSTARR analysis: https://colab.research.google.com/drive/1a2fiRPBd1xvoJf0WNiMUgTYiLTs1XETf
"""
Dataloader (implemented in Tensorflow) for applying evolution-inspired data augmentations
during training using tf.data.Dataset.
"""

import tensorflow as tf


class EvoAugDataset:
    """TensorFlow tf.data.Dataset wrapper for applying evolution-inspired data augmentations

    Parameters
    ----------
    x : tf.Tensor or numpy.ndarray
        Input sequences with shape (N, L, A).
    y : tf.Tensor or numpy.ndarray, optional
        Target labels. If None, only input data will be returned.
    augment_list : list
        List of data augmentations, each a callable class from augment.py.
        Default is empty list -- no augmentations.
    batch_size : int
        Batch size for the dataset, default is 32.
    max_augs_per_seq : int
        Maximum number of augmentations to apply to each sequence, default is 2.
    hard_aug : bool
        Flag to set a hard number of augmentations, otherwise the number of augmentations 
        is set randomly up to max_augs_per_seq, default is False.
    shuffle : bool
        Whether to shuffle the dataset, default is True.
    buffer_size : int
        Buffer size for shuffling, default is 1000.
    prefetch_size : int
        Number of batches to prefetch, default is tf.data.AUTOTUNE.
    apply_augmentations : bool
        Whether to apply augmentations (useful for validation sets), default is True.
    """
    
    def __init__(self, x, y=None, augment_list=[], batch_size=32, max_augs_per_seq=2, 
                 hard_aug=False, shuffle=True, buffer_size=1000, prefetch_size=tf.data.AUTOTUNE,
                 apply_augmentations=True):
        self.x = tf.convert_to_tensor(x, dtype=tf.float32)
        self.y = tf.convert_to_tensor(y, dtype=tf.float32) if y is not None else None
        self.augment_list = augment_list
        self.batch_size = batch_size
        self.max_augs_per_seq = min(max_augs_per_seq, len(augment_list)) if augment_list else 0
        self.hard_aug = hard_aug
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.prefetch_size = prefetch_size
        self.apply_augmentations = apply_augmentations
        self.max_num_aug = len(augment_list)
        self.insert_max = augment_max_len(augment_list)
        
        # Build the dataset
        self._dataset = self._build_dataset()

    def _build_dataset(self):
        """Build the tf.data.Dataset with augmentations."""
        if self.y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(self.x)
            
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
            
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        
        if self.apply_augmentations and self.augment_list:
            dataset = dataset.map(self._apply_augment_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
            
        dataset = dataset.prefetch(self.prefetch_size)
        return dataset

    def _apply_augment_wrapper(self, *args):
        """Wrapper for applying augmentations that handles both (x, y) and (x,) cases."""
        if len(args) == 2:
            x, y = args
            x_aug = self._apply_augment(x)
            return x_aug, y
        else:
            x = args[0]
            return self._apply_augment(x)


    @tf.function
    def _apply_augment(self, x):
        """Apply augmentations to each sequence in batch, x."""
        if not self.augment_list:
            return x
            
        # number of augmentations per sequence
        if self.hard_aug:
            batch_num_aug = tf.constant(self.max_augs_per_seq, dtype=tf.int32)
        else:
            batch_num_aug = tf.random.uniform(shape=[], minval=1, maxval=self.max_augs_per_seq+1, dtype=tf.int32)

        # randomly choose which subset of augmentations from augment_list
        aug_indices = tf.sort(tf.random.shuffle(tf.range(self.max_num_aug))[:batch_num_aug])
        
        # apply augmentation combination to sequences
        ind = 0
        for augment in self.augment_list:
            augment_condition = tf.reduce_any(tf.equal(tf.constant(ind), aug_indices))
            x = tf.cond(augment_condition, lambda aug=augment: aug(x), lambda: x)
            ind += 1
            
        return x


    def __iter__(self):
        """Make the dataset iterable."""
        return iter(self._dataset)

    def __len__(self):
        """Return the number of batches in the dataset."""
        return tf.data.experimental.cardinality(self._dataset).numpy()

    @property
    def dataset(self):
        """Get the underlying tf.data.Dataset object."""
        return self._dataset

    @classmethod
    def create_train_dataset(cls, x_train, y_train, augment_list, batch_size=32, **kwargs):
        """Create a training dataset with augmentations enabled."""
        return cls(x_train, y_train, augment_list=augment_list, batch_size=batch_size, 
                  apply_augmentations=True, shuffle=True, **kwargs)

    @classmethod 
    def create_val_dataset(cls, x_val, y_val, augment_list=None, batch_size=32, **kwargs):
        """Create a validation dataset without augmentations."""
        return cls(x_val, y_val, augment_list=augment_list or [], batch_size=batch_size,
                  apply_augmentations=False, shuffle=False, **kwargs)


#------------------------------------------------------------------------
# Helper function
#------------------------------------------------------------------------


def augment_max_len(augment_list):
    """Determine the maximum sequence length extension from augmentations.
    Since insertions now maintain sequence length by trimming, this returns 0.
    Kept for backward compatibility.
    
    Parameters
    ----------
    augment_list : list
        List of augmentations (unused since insertions no longer extend length).
    Returns
    -------
    int
        Value for max length extension (now always 0).
    """
    # augment_list parameter kept for backward compatibility but not used
    # since insertions now maintain original sequence length
    return 0
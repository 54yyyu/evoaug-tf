"""
Library of data augmentations for genomic sequence data.
All tensors are assumed to be one-hot encoded with shape (N, L, A):
    - N = batch size
    - L = sequence length
    - A = alphabet size (usually 4 for DNA)

To contribute a custom augmentation, use the following syntax:

.. code-block:: python

    class CustomAugmentation(AugmentBase):
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

        def __call__(self, x: tensorflow.Tensor) -> tensorflow.Tensor:
            # Perform augmentation
            return x_aug

"""

import tensorflow as tf

class AugmentBase():
    """Base class for EvoAug augmentation for genomic sequences."""
    
    def __call__(self, x):
        raise NotImplementedError()


class RandomTranslocation(AugmentBase):
    """Randomly cuts sequence in two pieces and shifts the order for each in a training 
    batch. This is implemented with a roll trasnformation with a user-defined shift_min 
    and shift_max. A different roll (positive or negative) is applied to each sequence. 
    Each sequence is padded with random DNA to ensure same shapes.

    Parameters
    ----------
    shift_min : int, optional
        Minimum size for random shift, defaults to 0.
    shift_max : int, optional
        Maximum size for random shift, defaults to 20.
    """
    def __init__(self, shift_min=0, shift_max=20):
        self.shift_min = shift_min
        self.shift_max = shift_max
    
    @tf.function
    def __call__(self, x):
        """Randomly shifts sequences in a batch, x.

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Sequences with random translocations.
        """
        N = tf.shape(x)[0]

        shifts = tf.random.uniform(
            shape=[N],
            minval=self.shift_min,
            maxval=self.shift_max + 1,
            dtype=tf.int32,
        )
        # randomize direction
        signs = tf.where(tf.random.uniform([N]) < 0.5, -1, 1)
        shifts = shifts * signs

        return tf.map_fn(lambda args: tf.roll(args[0], shift=args[1], axis=1),
                         (x, shifts),
                         fn_output_signature=tf.TensorSpec(shape=(None, None), dtype=x.dtype))


class RandomMutation(AugmentBase):
    """Randomly mutates sequences in a training batch according to a user-defined
    mutate_frac. A different set of mutations is applied to each sequence.

    Parameters
    ----------
    mutate_frac: float, optional
        Probability of mutation for each nucleotide, defaults to 0.05.
    """
    def __init__(self, mutate_frac=0.05):
        self.mutate_frac = mutate_frac
    
    @tf.function
    def __call__(self, x):
        """Randomly introduces mutations to a set of one-hot DNA sequences. 

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Sequences with randomly mutated DNA.
        """
        N = tf.shape(x)[0]
        L = tf.shape(x)[1]
        A = tf.shape(x)[2]

        # mask of sites to mutate
        mask = tf.random.uniform((N, L)) < self.mutate_frac

        # sample new nucleotides uniformly
        new_idx = tf.random.uniform((N, L), minval=0, maxval=A, dtype=tf.int32)
        new_onehot = tf.one_hot(new_idx, A, dtype=x.dtype)

        # replace where mask is True
        mask = tf.cast(mask[:, :, None], x.dtype)
        x_mut = (1 - mask) * x + mask * new_onehot
        return x_mut


class RandomInsertion(AugmentBase):
    """Randomly inserts a contiguous stretch of nucleotides into sequences in a training 
    batch according to a random number between a user-defined insert_min and insert_max. 
    A different insertion is applied to each sequence.

    Parameters
    ----------
    insert_min : int, optional
        Minimum size for random insertion, defaults to 0
    insert_max : int, optional
        Maximum size for random insertion, defaults to 20
    """
    def __init__(self, insert_min=0, insert_max=20):
        self.insert_min = insert_min
        self.insert_max = insert_max
    
    @tf.function
    def __call__(self, x):
        """Randomly inserts segments of random DNA into sequences while maintaining length.

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Sequences with randomly inserted segments of random DNA.
        """
        N, L, A = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Determine insertion length with proper bounds checking
        max_insert = tf.minimum(self.insert_max, L // 2)
        
        # Ensure minval < maxval for tf.random.uniform
        effective_min = tf.minimum(self.insert_min, max_insert)
        effective_max = tf.maximum(effective_min, max_insert)
        
        insert_len = tf.random.uniform([], effective_min, effective_max + 1, dtype=tf.int32)

        def do_insertion():
            # Generate different insertion positions for each sequence
            insert_starts = tf.random.uniform((N,), 0, L - insert_len + 1, dtype=tf.int32)
            
            # Generate random insertion DNA for all sequences
            new_idx = tf.random.uniform((N, L), minval=0, maxval=A, dtype=tf.int32)
            insertion = tf.one_hot(new_idx, A, dtype=x.dtype)
            
            # Create insertion mask for each sequence
            positions = tf.range(L)[None, :]  # (1, L)
            insert_starts_expanded = insert_starts[:, None]  # (N, 1)
            
            insert_mask = tf.logical_and(
                positions >= insert_starts_expanded,
                positions < insert_starts_expanded + insert_len
            )  # (N, L)
            
            # Expand to full shape
            insert_mask = insert_mask[:, :, None]  # (N, L, 1)
            insert_mask = tf.tile(insert_mask, [1, 1, A])  # (N, L, A)
            
            # Apply mask
            result = tf.where(insert_mask, insertion, x)
            return result

        return tf.cond(insert_len <= 0, lambda: x, do_insertion)

class RandomDeletion(AugmentBase):
    """Randomly deletes a contiguous stretch of nucleotides from sequences in a training 
    batch according to a random number between a user-defined delete_min and delete_max. 
    A different deletion is applied to each sequence.

    Parameters
    ----------
    delete_min : int, optional
        Minimum size for random deletion (defaults to 0). 
    delete_max : int, optional
        Maximum size for random deletion (defaults to 20). 
    """
    def __init__(self, delete_min=0, delete_max=20):
        self.delete_min = delete_min
        self.delete_max = delete_max
    
    @tf.function
    def __call__(self, x):
        """Randomly delete segments in a set of one-hot DNA sequences.

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Sequences with randomly deleted segments replaced with random DNA.
        """
        N, L, A = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Determine deletion length with proper bounds checking
        max_delete = tf.minimum(self.delete_max, L // 2)
        
        # Ensure minval < maxval for tf.random.uniform
        effective_min = tf.minimum(self.delete_min, max_delete)
        effective_max = tf.maximum(effective_min, max_delete)
        
        delete_len = tf.random.uniform([], effective_min, effective_max + 1, dtype=tf.int32)

        def do_deletion():
            # Generate different deletion positions for each sequence
            delete_starts = tf.random.uniform((N,), 0, L - delete_len + 1, dtype=tf.int32)
            
            # Generate random replacement DNA for all sequences
            new_idx = tf.random.uniform((N, L), minval=0, maxval=A, dtype=tf.int32)
            replacement = tf.one_hot(new_idx, A, dtype=x.dtype)
            
            # Create deletion mask for each sequence
            positions = tf.range(L)[None, :]  # (1, L)
            delete_starts_expanded = delete_starts[:, None]  # (N, 1)
            
            delete_mask = tf.logical_and(
                positions >= delete_starts_expanded,
                positions < delete_starts_expanded + delete_len
            )  # (N, L)
            
            # Expand to full shape
            delete_mask = delete_mask[:, :, None]  # (N, L, 1)
            delete_mask = tf.tile(delete_mask, [1, 1, A])  # (N, L, A)
            
            # Apply mask
            result = tf.where(delete_mask, replacement, x)
            return result

        return tf.cond(delete_len <= 0, lambda: x, do_deletion)



class RandomRC(AugmentBase):
    """Randomly applies a reverse-complement transformation to each sequence in a training 
    batch according to a user-defined probability, rc_prob. This is applied to each sequence 
    independently.

    Parameters
    ----------
    rc_prob: float, optional
        Probability to apply a reverse-complement transformation, defaults to 0.5.
    """
    
    def __init__(self, rc_prob=0.5):
        """Creates random reverse-complement object usable by Evoaug.
        """
        self.rc_prob = tf.constant(rc_prob)
    
    @tf.function
    def __call__(self, x):
        """ Randomly transforms sequences in a batch with a reverse-complement transformation. 

        Parameters
        ----------
        x : tf.tensor
            Batch of one-hot sequences (shape: (N, L, A))
        
        Returns
        -------
        tf.tensor
            Sequences with random reverse-complements applied.
        """
        N = tf.shape(x)[0]
        rc_mask = tf.random.uniform((N,)) < self.rc_prob

        # complement: assume A=0,C=1,G=2,T=3
        complement = tf.gather(x, [3, 2, 1, 0], axis=2)
        rc = tf.reverse(complement, axis=[1])

        return tf.where(rc_mask[:, None, None], rc, x)



class RandomNoise(AugmentBase):
    """Randomly add Gaussian noise to a batch of sequences with according to a use-defined
    noise_mean and noise_std. A different set of noise is applied to each sequence. 

    Parameters
    ----------
    noise_mean : float, optional
        Mean of the Gaussian noise, defaults to 0.0.
    noise_std : float, optional
        Standard deviation of the Gaussian noise, defaults to 0.2.
    """
    def __init__(self, noise_mean=0.0, noise_std=0.2):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
    
    @tf.function
    def __call__(self, x):
        """Randomly adds Gaussian noise to a set of one-hot DNA sequences.

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Sequences with random noise. 
        """
        return x + tf.random.normal(tf.shape(x), mean=self.noise_mean, stddev=self.noise_std)



#-----------------------------------------------------------------------------
# Batch mode augmentations
#-----------------------------------------------------------------------------


class RandomInsertionBatch(AugmentBase):
    """Randomly inserts a contiguous stretch of nucleotides into sequences in a training 
    batch according to a random number between a user-defined insert_min and insert_max. 
    The sequence length is maintained by replacing part of the sequence.
    The same insertion is applied to all sequences in the batch.

    Parameters
    ----------
    insert_min : int, optional
        Minimum size for random insertion, defaults to 0
    insert_max : int, optional
        Maximum size for random insertion, defaults to 20
    """
    def __init__(self, insert_min=0, insert_max=20):
        self.insert_min = insert_min
        self.insert_max = insert_max
    
    @tf.function
    def __call__(self, x):
        """Randomly inserts segments of random DNA into sequences while maintaining length.

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Sequences with randomly inserted segments of random DNA. Original sequence 
            length is maintained by replacing existing sequence.
        """
        N, L, A = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Determine insertion length with proper bounds checking
        max_insert = tf.minimum(self.insert_max, L // 2)
        
        # Ensure minval < maxval for tf.random.uniform
        effective_min = tf.minimum(self.insert_min, max_insert)
        effective_max = tf.maximum(effective_min, max_insert)
        
        insert_len = tf.random.uniform([], effective_min, effective_max + 1, dtype=tf.int32)

        def do_insertion():
            # Choose insertion start position
            insert_start = tf.random.uniform([], 0, L - insert_len + 1, dtype=tf.int32)
            
            # Generate random insertion DNA for all sequences
            new_idx = tf.random.uniform((N, L), minval=0, maxval=A, dtype=tf.int32)
            insertion = tf.one_hot(new_idx, A, dtype=x.dtype)
            
            # Create insertion mask - True where we should use insertion DNA
            positions = tf.range(L)  # (L,)
            insert_mask = tf.logical_and(
                positions >= insert_start,
                positions < insert_start + insert_len
            )  # (L,)
            
            # Expand mask to full tensor shape
            insert_mask = insert_mask[None, :, None]  # (1, L, 1)
            insert_mask = tf.tile(insert_mask, [N, 1, A])  # (N, L, A)
            
            # Apply mask: use insertion where mask is True, original where False
            result = tf.where(insert_mask, insertion, x)
            return result

        return tf.cond(insert_len <= 0, lambda: x, do_insertion)




class RandomDeletionBatch(AugmentBase):
    """Randomly deletes a contiguous stretch of nucleotides from sequences in a training 
    batch according to a random number between a user-defined delete_min and delete_max. 
    The same deletion is applied to all sequences in the batch.
    
    This version uses masking to avoid shape inference issues.

    Parameters
    ----------
    delete_min : int, optional
        Minimum size for random deletion (defaults to 0). 
    delete_max : int, optional
        Maximum size for random deletion (defaults to 20). 
    """
    def __init__(self, delete_min=0, delete_max=20):
        self.delete_min = delete_min
        self.delete_max = delete_max
    
    @tf.function
    def __call__(self, x):
        """Randomly delete segments in a set of one-hot DNA sequences. 

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Sequences with randomly deleted segments replaced with random DNA.
            Original shape is preserved.
        """
        N, L, A = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Determine deletion length with proper bounds checking
        max_delete = tf.minimum(self.delete_max, L // 2)
        
        # Ensure minval < maxval for tf.random.uniform
        effective_min = tf.minimum(self.delete_min, max_delete)
        effective_max = tf.maximum(effective_min, max_delete)
        
        delete_len = tf.random.uniform([], effective_min, effective_max + 1, dtype=tf.int32)

        def do_deletion():
            # Choose deletion start position
            delete_start = tf.random.uniform([], 0, L - delete_len + 1, dtype=tf.int32)
            
            # Generate random replacement DNA for all sequences
            new_idx = tf.random.uniform((N, L), minval=0, maxval=A, dtype=tf.int32)
            replacement = tf.one_hot(new_idx, A, dtype=x.dtype)
            
            # Create deletion mask - True where we should use replacement DNA
            positions = tf.range(L)  # (L,)
            delete_mask = tf.logical_and(
                positions >= delete_start,
                positions < delete_start + delete_len
            )  # (L,)
            
            # Expand mask to full tensor shape
            delete_mask = delete_mask[None, :, None]  # (1, L, 1)
            delete_mask = tf.tile(delete_mask, [N, 1, A])  # (N, L, A)
            
            # Apply mask: use replacement where mask is True, original where False
            result = tf.where(delete_mask, replacement, x)
            return result

        return tf.cond(delete_len <= 0, lambda: x, do_deletion)




class RandomTranslocationBatch(AugmentBase):
    """Randomly cuts sequence in two pieces and shifts the order for each in a training 
    batch. This is implemented with a roll trasnformation with a user-defined shift_min 
    and shift_max. A different roll (positive or negative) is applied to each sequence. 
    Each sequence is padded with random DNA to ensure same shapes.

    Parameters
    ----------
    shift_max : int, optional
        Maximum size for random shift, defaults to 20.
    """
    def __init__(self, shift_min=0, shift_max=20):
        self.shift_min = shift_min
        self.shift_max = shift_max
    
    @tf.function
    def __call__(self, x):
        """Randomly shifts sequences in a batch, x.

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Sequences with random translocations.
        """
        shift = tf.random.uniform([], minval=self.shift_min,
                                  maxval=self.shift_max + 1, dtype=tf.int32)
        if tf.random.uniform([]) < 0.5:
            shift = -shift
        return tf.roll(x, shift=shift, axis=1)




class RandomRCBatch(AugmentBase):
    """Randomly applies a reverse-complement transformation to each sequence in a training 
    batch according to a user-defined probability, rc_prob. This is applied to each sequence 
    independently.

    Parameters
    ----------
    rc_prob: float, optional
        Probability to apply a reverse-complement transformation, defaults to 0.5.
    """
    def __init__(self, rc_prob=0.5):
        """Creates random reverse-complement object usable by Evoaug.
        """
        self.rc_prob = tf.constant(rc_prob)

    @tf.function
    def __call__(self, x):
        """ Randomly transforms sequences in a batch with a reverse-complement transformation. 

        Parameters
        ----------
        x : tf.tensor
            Batch of one-hot sequences (shape: (N, L, A))
        
        Returns
        -------
        tf.tensor
            Sequences with random reverse-complements applied.
        """
        apply = tf.random.uniform([]) < self.rc_prob
        complement = tf.gather(x, [3, 2, 1, 0], axis=2)
        rc = tf.reverse(complement, axis=[1])
        return tf.cond(apply, lambda: rc, lambda: x)






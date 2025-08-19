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
    The sequence length is maintained by trimming the end of the sequence after insertion.
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
            Sequences with randomly inserted segments of random DNA. Original sequence 
            length is maintained by trimming the end.
        """
        N, L, A = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        insert_len = tf.random.uniform([], self.insert_min,
                                       tf.minimum(self.insert_max, L // 2) + 1,
                                       dtype=tf.int32)

        def do_insertion():
            # Generate random DNA insertions for each sequence
            new_idx = tf.random.uniform((N, insert_len), minval=0, maxval=A, dtype=tf.int32)
            insertions = tf.one_hot(new_idx, A, dtype=x.dtype)

            # Generate random insertion positions for each sequence
            insert_inds = tf.random.uniform((N,), minval=0, maxval=L - insert_len + 1, dtype=tf.int32)

            # Create output tensor
            result = tf.zeros_like(x)
            
            # Use tf.while_loop to process each sequence
            def cond(i, result):
                return i < N
            
            def body(i, result):
                seq = x[i]
                insertion = insertions[i]
                insert_ind = insert_inds[i]
                
                # Use tf.cond to handle the slicing
                def insert_at_position():
                    before = seq[:insert_ind]
                    after = seq[insert_ind:L-insert_len]
                    new_seq = tf.concat([before, insertion, after], axis=0)
                    return new_seq
                
                def insert_at_zero():
                    after = seq[:L-insert_len]
                    new_seq = tf.concat([insertion, after], axis=0)
                    return new_seq
                
                new_seq = tf.cond(insert_ind > 0, insert_at_position, insert_at_zero)
                result = tf.tensor_scatter_nd_update(result, [[i]], [new_seq])
                return i + 1, result
            
            _, result = tf.while_loop(cond, body, [0, result])
            return result

        return tf.cond(insert_len <= 0, lambda: x, do_insertion)

class RandomDeletion(AugmentBase):
    """Randomly deletes a contiguous stretch of nucleotides from sequences in a training 
    batch according to a random number between a user-defined delete_min and delete_max. 
    The sequence length is maintained by padding with random DNA. A different deletion 
    is applied to each sequence.

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
        """Randomly delete segments in a set of one-hot DNA sequences while maintaining length.

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Sequences with randomly deleted segments, padded with random DNA to maintain 
            original sequence length.
        """
        N, L, A = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        delete_len = tf.random.uniform([], self.delete_min,
                                       tf.minimum(self.delete_max, L // 2) + 1,
                                       dtype=tf.int32)

        def do_deletion():
            # Generate random DNA padding for each sequence
            new_idx = tf.random.uniform((N, delete_len), minval=0, maxval=A, dtype=tf.int32)
            padding = tf.one_hot(new_idx, A, dtype=x.dtype)

            # Generate random deletion positions for each sequence
            delete_inds = tf.random.uniform((N,), minval=0, maxval=L - delete_len + 1, dtype=tf.int32)

            # Create output tensor
            result = tf.zeros_like(x)
            
            # Use tf.while_loop to process each sequence
            def cond(i, result):
                return i < N
            
            def body(i, result):
                seq = x[i]
                pad = padding[i]
                delete_ind = delete_inds[i]
                
                # Use tf.cond to handle the slicing
                def delete_at_position():
                    before = seq[:delete_ind]
                    after = seq[delete_ind + delete_len:]
                    new_seq = tf.concat([before, after, pad], axis=0)
                    return new_seq
                
                def delete_at_zero():
                    after = seq[delete_len:]
                    new_seq = tf.concat([after, pad], axis=0)
                    return new_seq
                
                new_seq = tf.cond(delete_ind > 0, delete_at_position, delete_at_zero)
                result = tf.tensor_scatter_nd_update(result, [[i]], [new_seq])
                return i + 1, result
            
            _, result = tf.while_loop(cond, body, [0, result])
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
    The sequence length is maintained by trimming the end of the sequence after insertion.
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
            length is maintained by trimming the end.
        """
        N, L, A = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        insert_len = tf.random.uniform([], self.insert_min,
                                       tf.minimum(self.insert_max, L // 2) + 1,
                                       dtype=tf.int32)

        def do_insertion():
            # Generate random DNA insertion (same for all sequences)
            new_idx = tf.random.uniform((insert_len,), minval=0, maxval=A, dtype=tf.int32)
            insertion = tf.one_hot(new_idx, A, dtype=x.dtype)
            insertion = tf.expand_dims(insertion, 0)  # (1, insert_len, A)
            insertion = tf.tile(insertion, [N, 1, 1])  # (N, insert_len, A)

            # Generate random insertion position (same for all sequences)
            insert_ind = tf.random.uniform([], minval=0, maxval=L - insert_len + 1, dtype=tf.int32)

            # Split all sequences at the same position
            before = x[:, :insert_ind, :]  # (N, insert_ind, A)
            after = x[:, insert_ind:L-insert_len, :]  # (N, L-insert_len-insert_ind, A)
            
            # Concatenate: before + insertion + after
            result = tf.concat([before, insertion, after], axis=1)
            return result

        return tf.cond(insert_len <= 0, lambda: x, do_insertion)




class RandomDeletionBatch(AugmentBase):
    """Randomly deletes a contiguous stretch of nucleotides from sequences in a training 
    batch according to a random number between a user-defined delete_min and delete_max. 
    The same deletion is applied to all sequences in the batch.

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
            Sequences with randomly deleted segments (padded to correct shape
            with random DNA)
        """
        N, L, A = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        delete_len = tf.random.uniform([], self.delete_min,
                                       tf.minimum(self.delete_max, L // 2) + 1,
                                       dtype=tf.int32)

        def do_deletion():
            # Generate random DNA padding (same for all sequences)
            new_idx = tf.random.uniform((delete_len,), minval=0, maxval=A, dtype=tf.int32)
            padding = tf.one_hot(new_idx, A, dtype=x.dtype)
            padding = tf.expand_dims(padding, 0)  # (1, delete_len, A)
            padding = tf.tile(padding, [N, 1, 1])  # (N, delete_len, A)

            # Generate random deletion position (same for all sequences)
            delete_ind = tf.random.uniform([], minval=0, maxval=L - delete_len + 1, dtype=tf.int32)

            # Split all sequences at the same position (skipping deleted region)
            before = x[:, :delete_ind, :]  # (N, delete_ind, A)
            after = x[:, delete_ind + delete_len:, :]  # (N, L-delete_ind-delete_len, A)
            
            # Concatenate: before + after + padding
            result = tf.concat([before, after, padding], axis=1)
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






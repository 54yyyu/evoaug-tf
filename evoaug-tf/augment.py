"""
Library of data augmentations for genomic sequence data.

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
    """
    Base class for EvoAug augmentation for genomic sequences.
    """
    def __call__(self, x, y=None):
        """Return an augmented version of 'x'.

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Batch of one-hot sequences with random augmentation applied.
        """
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

        # determine size of shifts for each sequence
        shifts = tf.random.uniform(shape=[N,], minval=-1*self.shift_max, maxval=self.shift_max, dtype=tf.int32)
        #shifts = tf.random.uniform(shape=(N,), minval=self.shift_min, maxval=self.shift_max + 1, dtype=tf.int32)


        # apply random shift to each sequence
        x_rolled = tf.TensorArray(dtype=x.dtype, size=N, element_shape=x[0].shape)
        body = lambda i, x_rolled: (i + 1, x_rolled.write(i, tf.roll(x[i], shift=shifts[i], axis=0)))
        cond = lambda i, x_rolled: i < tf.shape(shifts)[0]
        _, x_rolled = tf.while_loop(cond, body, [0, x_rolled])
        x_new = x_rolled.stack()

        return x_new


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
        x : torch.Tensor
            Batch of one-hot sequences (shape: (N, A, L)).
        
        Returns
        -------
        torch.Tensor
            Sequences with randomly mutated DNA.
        """
        N = tf.shape(x)[0]
        L = tf.shape(x)[1]
        A = tf.cast(tf.shape(x)[2], dtype = tf.float32)

        # determine the number of mutations per sequence
        num_mutations = tf.cast(tf.round(tf.cast(self.mutate_frac / 0.75, dtype=tf.float32) * tf.cast(L, dtype=tf.float32)), dtype=tf.int32)

        # randomly determine the indices to apply mutations
        mutation_inds = tf.slice(tf.argsort(tf.random.uniform(shape=(N, L)), axis=1), [0,0], [N,num_mutations])

        a = tf.eye(A)
        p = tf.ones((A,)) / A
        mutations = tf.transpose(tf.gather(a, tf.random.categorical(tf.math.log(tf.repeat([p], repeats=num_mutations, axis=0)), N)), perm=[1,0,2])

        x_aug = tf.TensorArray(x.dtype, size=N)

        i = tf.constant(0)

        while_condition = lambda i, _: tf.less(i, N)

        body = lambda i, x_aug: (
            i + 1, 
            x_aug.write(i, tf.tensor_scatter_nd_update(x[i], tf.expand_dims(mutation_inds[i], axis=1), mutations[i]))
        )

        _, x_aug = tf.while_loop(while_condition, body, loop_vars=[i, x_aug])
        x_rolled = x_aug.stack()
        return x_rolled


class RandomInsertion(AugmentBase):
    """Randomly inserts a contiguous stretch of nucleotides from sequences in a training 
    batch according to a random number between a user-defined insert_min and insert_max. 
    A different insertoins is applied to each sequence. Each sequence is padded with random 
    DNA to ensure same shapes.

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
    def __call__(self,x):
        """Randomly inserts segments of random DNA to a set of DNA sequences. 

        Parameters
        ----------
        x : tf.Tensor
            Batch of one-hot sequences (shape: (N, L, A)).
        
        Returns
        -------
        tf.Tensor
            Sequences with randomly inserts segments of random DNA. All sequences 
            are padded with random DNA to ensure same shape. 
        """
        N = tf.shape(x)[0]
        L = tf.shape(x)[1]
        A = tf.cast(tf.shape(x)[2], dtype = tf.float32)

        # sample random DNA
        a = tf.eye(A)
        p = tf.ones((A,)) / A
        insertions = tf.transpose(tf.gather(a, tf.random.categorical(tf.math.log([p] * self.insert_max), N)), perm=[1,0,2])

        # sample insertion length for each sequence
        insert_lens = tf.random.uniform((N,), minval=self.insert_min, maxval=self.insert_max + 1, dtype=tf.int32)

        # sample locations to insertion for each sequence
        insert_inds = tf.random.uniform((N,), minval=0, maxval=L, dtype=tf.int32)

        # loop over each sequence
        i = tf.constant(0)
        x_aug = tf.TensorArray(x.dtype, size=N)
    
        while_condition = lambda i, _: tf.less(i, N)

        body = lambda i, x_aug: (
            i + 1, 
            x_aug.write(i, tf.concat([insertions[i][:tf.math.floordiv((self.insert_max - insert_lens[i]), 2), :],                                                                                   # random dna padding
                                                x[i][:insert_inds[i], :],                                                                                                                           # sequence up to insertoin start index
                                                insertions[i][tf.math.floordiv((self.insert_max - insert_lens[i]), 2):tf.math.floordiv((self.insert_max - insert_lens[i]), 2)+insert_lens[i], :],   # random insertion
                                                x[i][insert_inds[i]:, :],                                                                                                                           # sequence after insertion end index
                                                insertions[i][tf.math.floordiv((self.insert_max - insert_lens[i]), 2)+insert_lens[i]:self.insert_max, :]],                                          # random dna padding
                                                axis=0))
        )


        _, x_aug = tf.while_loop(while_condition, body, loop_vars=[i, x_aug])
        x_rolled = x_aug.stack()
        return x_rolled


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
            Sequences with randomly deleted segments (padded to correcct shape
            with random DNA)
        """
        N = tf.shape(x)[0]
        L = tf.shape(x)[1]
        A = tf.cast(tf.shape(x)[2], dtype = tf.float32)

        # sample random DNA
        a = tf.eye(A)
        p = tf.ones((A,)) / A
        padding = tf.transpose(tf.gather(a, tf.random.categorical(tf.math.log([p] * self.delete_max), N)), perm=[1,0,2])

        # sample deletion length for each sequence
        delete_lens = tf.random.uniform((N,), minval=self.delete_min, maxval=self.delete_max + 1, dtype=tf.int32)

        # sample locations to delete for each sequence
        delete_inds = tf.random.uniform((N,), minval=0, maxval=L - self.delete_max + 1, dtype=tf.int32)

        # loop over each sequence
        i = tf.constant(0)
        x_aug = tf.TensorArray(x.dtype, size=N)
        while_condition = lambda i, _: tf.less(i, N)

        body = lambda i, x_aug: (
            i + 1, 
            x_aug.write(i, tf.concat([padding[i][:tf.math.floordiv(delete_lens[i], 2), :],                                                  # random dna padding
                                                x[i][:delete_inds[i], :],                                                                   # sequence up to deletion start index
                                                x[i][delete_inds[i]+delete_lens[i]:, :],                                                    # sequence after deletion end index
                                                padding[i][self.delete_max-delete_lens[i]+tf.math.floordiv(delete_lens[i], 2):, :]],        # random dna padding
                                                axis=0))                                                                                    # concatenation axis
        )

        _, x_aug = tf.while_loop(while_condition, body, loop_vars=[i, x_aug])
        return x_aug.stack()


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
        """ Randomly transforms sequences in a batch with a reverse-compleemnt transformation. 

        Parameters
        ----------
        x : tf.tensor
            Batch of one-hot sequences (shape: (N, L, A))
        
        Returns
        -------
        tf.tensor
            Sequences with random reverse-complements applied.
        """
        # make a copy of the sequence
        x_aug = tf.identity(x)

        # randomly select sequences to apply rc transformation
        ind_rc = tf.random.uniform(shape=[tf.shape(x)[0],]) < self.rc_prob

        # apply reverse-compement transformation
        x_new = tf.where(ind_rc[:, None, None], tf.reverse(x_aug, axis=[1,2]), x_aug)
        return x_new


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
        return x + tf.random.normal(shape=tf.shape(x), mean=self.noise_mean, stddev=self.noise_std)

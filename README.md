# EvoAug-TF
Evolution-inspired data augmentations for TensorFlow-based models for regulatory genomics

#### Install:

```
pip install git+https://github.com/54yyyu/evoaug_tf.git
```


#### Dependencies:

```
tensorflow 2.11.0+cu114
numpy 1.21.6
```

#### Example

```python
from evoaug_tf import evoaug_tf, augment
import tensorflow as tf
from tensorflow import keras

model = "DEFINE KERAS MODEL"

augment_list = [
]

robust_model = evoaug_tf.RobustModel(model, augment_list=augment_list, max_augs_per_seq=2, hard_aug=True, inference_aug=False)

robust_model.compile()

robust_model.fit(x_train, y_train)
```

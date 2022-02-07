# MNIST challenge

## Training

For the classifier I used a simple 2 layer network as shown in the tensorflow MNIST examples. 

1. Put the training / test data in a 'training/data/' directory
2. Run the training container while in the root directory of the project

```commandline
mkdir models

docker build -t mnist-training training/

docker run --rm \
--mount type=bind,source=/path/to/project/models/,target=/models/ \
mnist-training
```

This model achieves sufficient accuracy for the challenge after a couple of epochs:
```commandline
Epoch 1/6
469/469 [==============================] - 5s 11ms/step - loss: 0.3615 - sparse_categorical_accuracy: 0.8999 - val_loss: 0.1976 - val_sparse_categorical_accuracy: 0.9422
Epoch 2/6
469/469 [==============================] - 1s 2ms/step - loss: 0.1687 - sparse_categorical_accuracy: 0.9523 - val_loss: 0.1408 - val_sparse_categorical_accuracy: 0.9586
Epoch 3/6
469/469 [==============================] - 1s 2ms/step - loss: 0.1218 - sparse_categorical_accuracy: 0.9659 - val_loss: 0.1112 - val_sparse_categorical_accuracy: 0.9672
Epoch 4/6
469/469 [==============================] - 1s 2ms/step - loss: 0.0945 - sparse_categorical_accuracy: 0.9732 - val_loss: 0.0953 - val_sparse_categorical_accuracy: 0.9710
Epoch 5/6
469/469 [==============================] - 1s 2ms/step - loss: 0.0764 - sparse_categorical_accuracy: 0.9780 - val_loss: 0.0865 - val_sparse_categorical_accuracy: 0.9731
Epoch 6/6
469/469 [==============================] - 1s 2ms/step - loss: 0.0632 - sparse_categorical_accuracy: 0.9822 - val_loss: 0.0810 - val_sparse_categorical_accuracy: 0.9755
```

## Scoring

For scoring I used the tensorflow/serving image. A model produced by the training step can be served to a
REST API endpoint with the following:
```commandline
docker pull tensorflow/serving

docker run --rm -p 8501:8501 \
  --mount type=bind,source=/path/to/project/models/mnist_clf/,target=/models/mnist_clf \
  -e MODEL_NAME=mnist_clf -t tensorflow/serving
```
The ``scoring/query_example.py`` script shows an example of how to get a prediction from the endpoint. Note that the
model uses a single colour channel.
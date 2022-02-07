import os

import tensorflow as tf
import tensorflow_datasets as tfds


def train_mnist(
				learning_rate: float = 0.001,
				batch_size: int = 128,
				layer_width: int = 128,
				training_epochs: int = 6,
				data_path: str = 'data/',
				model_path: str = 'models/',
				model_version: int = 1):
	"""
	Trains a basic MNIST classifier and saves it for inference with tensorflow serving.

	The model is trained using only a single colour channel as input.

	:param learning_rate:		Training learning rate.
	:param batch_size:			Number of samples used for each SGD update.
	:param layer_width:			Width of the hidden layer.
	:param training_epochs:	Number of times to iterate over the entire training set.
	:param data_path:				Path to folder container 'training' and 'testing' directories.
	:param model_path:			Path to save trained model to.
	:param model_version:		Version number to tag trained model with.
	"""
	builder = tfds.ImageFolder(data_path)

	ds_train = builder.as_dataset(split='training', shuffle_files=True, as_supervised=True)
	ds_test = builder.as_dataset(split='testing', as_supervised=True)

	# Cache after preprocessing
	normalization_layer = tf.keras.layers.Rescaling(1. / 255)
	ds_train = ds_train.map(lambda x, y: (normalization_layer(x)[:, :, 0], y))
	ds_train = ds_train.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

	ds_test = ds_test.map(lambda x, y: (normalization_layer(x)[:, :, 0], y))
	ds_test = ds_test.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

	model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(input_shape=(28, 28)),
		tf.keras.layers.Dense(layer_width, activation='relu'),
		tf.keras.layers.Dense(10)])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

	model.fit(ds_train, epochs=training_epochs, validation_data=ds_test)

	if not os.path.exists(model_path):
		os.mkdir(model_path)

	# Inference function for tensorflow/serving
	@tf.function(input_signature=[tf.TensorSpec([None, 28, 28], dtype=tf.float32)])
	def inference_function(model_input):
		model_input = model_input / 255.

		# inference model
		logits = model(model_input)

		outputs = tf.math.argmax(tf.nn.softmax(logits), axis=1)
		return outputs

	tf.saved_model.save(
		model,
		os.path.join(model_path, 'mnist_clf', str(model_version)),
		signatures={
				"serving_default": inference_function,
			}
	)


if __name__ == '__main__':
	train_mnist()

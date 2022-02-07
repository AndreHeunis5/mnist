import tensorflow_datasets as tfds
import json
import requests
import numpy as np

if __name__ == "__main__":
	builder = tfds.ImageFolder('../training/data/')
	ds_test = builder.as_dataset(split='testing', as_supervised=True)

	image_batch = next(iter(ds_test))
	first_image = image_batch[0]

	# Model uses single channel for prediction
	test_np = np.array(first_image)[:, :, 0]
	print(test_np.shape)

	data = json.dumps({"signature_name": "serving_default", "instances": test_np.tolist()})

	headers = {"content-type": "application/json"}
	json_response = requests.post('http://localhost:8501/v1/models/mnist_clf:predict', data=data, headers=headers)

	print('JSON response: ', json_response)

	predictions = json.loads(json_response.text)
	print('PREDICTION: ', predictions)
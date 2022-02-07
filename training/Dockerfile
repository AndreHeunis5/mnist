#FROM tensorflow/tensorflow
FROM python:3.8-slim-buster

COPY requirements.txt train.py ./
ADD data/ data
RUN pip install -r requirements.txt && rm requirements.txt

#COPY train.py .

CMD ["python", "train.py"]
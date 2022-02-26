
FROM python:latest

# Need to import this here instead of the requirments doc 
FROM tensorflow/tensorflow:latest-gpu

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set work directory.
RUN mkdir /developer

WORKDIR /developer

COPY requirements.txt /developer/

RUN pip install -r requirements.txt
# RUN pip install opencv-python
# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . /developer/

EXPOSE 5000

# Need to import this here instead of the requirments doc 
FROM tensorflow/tensorflow:latest-gpu

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set work directory.
RUN mkdir /developer

WORKDIR /developer

COPY requirements.txt /developer/

RUN pip install -r requirements.txt

# RUN  apt-get update \
#   && apt-get install -y wget
# RUN wget "https://raw.githubusercontent.com/NVIDIA/TensorRT/main/quickstart/IntroNotebooks/helper.py"
# RUN pip install opencv-python
# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install nvidia-pyindex && pip install nvidia-tensorrt

COPY . /developer/

EXPOSE 5000
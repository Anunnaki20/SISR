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

COPY . /developer/

EXPOSE 8080
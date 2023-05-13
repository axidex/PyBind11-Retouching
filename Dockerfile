# Don't forget about "$ xhost +" before running container
FROM ubuntu:latest

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Installing linux libs
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    python3.10-dev \
    python3-pip
RUN cp /usr/lib/gcc/x86_64-linux-gnu/11/crtbeginS.o /usr/lib/gcc/x86_64-linux-gnu/11/crtbeginT.o
# Installing python libs
RUN pip install numpy \
    opencv-python

# Copying some data
COPY . /app
WORKDIR /app

RUN git clone https://github.com/davisking/dlib external/dlib
RUN git clone https://github.com/pybind/pybind11 external/pybind11
RUN git clone https://github.com/msteinbeck/tinyspline external/tinyspline
RUN git clone https://github.com/axidex/wavelib external/wavelib

# Building and running
RUN mkdir build
WORKDIR /app/build

RUN cp ../assets/shape_predictor_68_face_landmarks.dat .
RUN cp ../assets/Orig1.png .
RUN cp ../assets/test.py .

RUN export DISPLAY=:0

RUN cmake ..
RUN make

RUN python3 test.py

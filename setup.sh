#!/bin/sh
wget=/usr/bin/wget
sudo apt-get update
sudo apt-get install python-pip
sudo pip install opencv-python==3.1.0.0
sudo pip install tensorflow
sudo apt-get install protobuf-compiler python-pil python-lxml
sudo apt-get install  python-tk
sudo pip install  matplotlib
sudo pip install -U scikit-learn
sudo pip install pandas
sudo apt-get install zip unzip
sudo apt-get install imagemagick
sudo chmod 777 /usr/local

cd lib
wget https://github.com/tensorflow/models/archive/master.zip
unzip master.zip
sudo rm master.zip
sudo cp -r models-master/research api
cd api 
protoc object_detection/protos/*.proto --python_out=.

cd ..
sudo rm -r models-master/
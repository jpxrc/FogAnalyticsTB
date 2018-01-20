#!/bin/sh
wget=/usr/bin/wget
sudo apt-get update
sudo apt-get install python-pip
sudo pip install opencv-python==3.1.0.0
sudo pip install tensorflow
sudo apt-get install protobuf-compiler python-pil python-lxml
sudo apt-get install  python-tk
sudo pip install  matplotlib
sudo apt-get install zip unzip

# wget https://github.com/junostar/FogAnalyticsTB/archive/master.zip
# #wget https://github.com/junostar/FogAnalyticsTB.zip
# unzip FogAnalyticsTB.zip
# sudo rm FogAnalyticsTB.zip
cd lib
wget https://github.com/tensorflow/models/archive/master.zip
unzip master.zip
rm master.zip
sudo cp -r models-master/research api
cd api 
protoc object_detection/protos/*.proto --python_out=.

cd ..
sudo rm -r models-master/
cd ..

python pipeline.py test


#wget --no-check-certificate https://github.com/junostar/FogAnalyticsTB/archive/master.tar.gz
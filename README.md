# 杂七杂八

## build   
git clone --recursive https://github.com/apache/incubator-tvm tvm    
git submodule init   
git submodule update     
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev     
mkdir build    
cp cmake/config.cmake build   

## env
export TVM_HOME=/path/to/tvm    
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}    


## reference

https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github   
https://github.com/apache/incubator-tvm    
https://github.com/htshinichi/caffe-onnx    
   

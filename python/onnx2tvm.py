import onnx
import numpy as np
import tvm
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import util, ndk, graph_runtime as runtime
from tvm.contrib.download import download_testdata
import os
from timeit import default_timer as timer
import datetime
from tvm.contrib.debugger import debug_runtime as graph_runtime

onnx_model = onnx.load('facenet.onnx')
#build for android
#target = 'llvm -target=armv7a-linux-android -mfloat-abi=soft -mattr=+neon'
#build for x86_64
target = 'llvm'
#"input.1"
input_name = 'input_input'
input = np.random.rand(1,3,112,112)
shape_dict = {input_name:input.shape}
model, params = relay.frontend.from_onnx(onnx_model,shape_dict)

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(model, target, params=params)


path_lib = "deploy/lib/facenet.so"
lib.export_library(path_lib)

fo=open("facenet.json","w")
fo.write(graph)
fo.close()
 
fo=open("facenet.params","wb")
fo.write(relay.save_param_dict(params))
fo.close()


#print(lib)
#relay.build_config(opt_level=3)
#graph, lib, params = relay.build(model, target, params=params)

"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
image_path = 'test.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image_mean = np.array([127.5, 127.5, 127.5])
image = (image - image_mean)/ 127.5
input_image = image.astype(dtype=np.float32)
input_image = input_image[:, :, :, np.newaxis]
input_image = input_image.transpose([3, 2, 0, 1])

#from tvm.contrib import graph_runtime

ctx = tvm.gpu(0)
m = graph_runtime.create(graph, lib, ctx)
dtype = 'float32'
#complete a inference
#print(input_image)
m.set_input("input.1", tvm.nd.array(input_image.astype(dtype)))
m.set_input(**params)
# set start time
#start = timer()

for i in range(6):
    m.set_input("input.1", tvm.nd.array(input_image.astype(dtype)))
 #  m.set_input(**params)
    m.run()
    tvm_output = m.get_output(0)
#    print(tvm_output)


print("start")
time_all = 0
start = datetime.datetime.now()
for i in range(100):
#    x = np.random.rand(1024,1024,3)
#    input_image = x[:, :, :, np.newaxis]
#    input_image = input_image.transpose([3, 2, 0, 1])
#     m.set_input("input.1", tvm.nd.array(input_image.astype(dtype)))
#    m.set_input(**params)
#    start = timer()
#    start_1 = timer()
     start_1 = datetime.datetime.now()
     m.run()
     tvm_output = m.get_output(0)
#    print(tvm_output)
#    end_1 = timer()
     end_1 = datetime.datetime.now()
#     print((end_1 - start_1).seconds*1000.0 + (end_1 - start_1).microseconds/1000.0)
#    print(end_1 - start_1)
#    end = timer()
#    time_all += (end-start)
#    tvm_output = m.get_output(0)
#    if i == 2:
#        end = timer()
#        timeuse_warmup = end - start
end = datetime.datetime.now()
print("end")
print(((end - start).seconds*1000.0 + (end - start).microseconds/1000.0)/100.0)
#print('warmup time: ' , timeuse_warmup)
#print('time: ' , (timeuse))
print(tvm_output.shape)


path_lib = "deploy/lib/resnet18_arm.so"
fcompile = ndk.create_shared
#lib.export_library(path_lib,fcompile)
lib.export_library(path_lib)

fo=open("resnet18.json","w")
fo.write(graph)
fo.close()

fo=open("resnet18.params","wb")
fo.write(relay.save_param_dict(params))
fo.close()
"""

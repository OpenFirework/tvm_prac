#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <math.h>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <sys/time.h>

#include <sys/time.h>
#include <stdio.h>

int32_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int32_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}


int main()
{
    printf("starting run.\n");
    // tvm module for compiled functions
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile("./resnet18_arm.so");
    printf("load lib.so ok.\n");

    // json graph
    std::ifstream json_in("./resnet18.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();
    printf("load graph ok.\n");

    // parameters in binary
    std::ifstream params_in("./resnet18.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();
    printf("load param ok.\n");

    //run model
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;

     // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);

    DLTensor* x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 224, 224 };
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    
      
    float data[224 * 224 * 3];
    int loop = 224*224*3;
    for(int i=0;i<loop;i++) {
      data[i] = i;
    }

    memcpy(x->data, &data, 3 * 224 * 224 * sizeof(float));
    printf("load data ok.\n");

    //time test
    int count = 0;
    DLTensor* y;
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
		load_params(params_arr);
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = mod.GetFunction("run");

     int out_ndim = 2;
		 int64_t out_shape[2] = {1, 1000};
		 TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);


    for(int i=0;i<5;i++) {  
    //  printf("set input data redy\n");
      set_input("input.1", x);  
		// get the function from the module(run it)
		//  printf("set input data down\n");
		  run();

     // int out_ndim = 4;
		//  int64_t out_shape[4] = {1,1,1,1000};
		 // TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
      
		  get_output(0, y);
   
    }
      
    int32_t t1 = NowMicros();
    for(int i=0;i<20;i++) {  

    set_input("input.1", x); 
		// get the function from the module(run it)
		 
		  run();

   
      
		  get_output(0, y);
      
   
    }
    int32_t t2 = NowMicros();
    printf("%dms\n",(t2-t1)/(20*1000));
    TVMArrayFree(x);
    TVMArrayFree(y);
    return 0;
}

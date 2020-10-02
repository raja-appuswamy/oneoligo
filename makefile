CXX=clang++
PARAMS=-DCUDA=1 -DNUM_STRING=450000
CXXFLAGS= -O2 -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda $(PARAMS)
LDFLAGS= -fsycl -lsycl -ltbb -lpthread
DPCPP_EXE_NAME=embedjoin

build:	src/embedjoin_dpcpp.cpp	src/verification.cpp
	$(CXX) $(CXXFLAGS) src/embedjoin_dpcpp.cpp src/verification.cpp $(LDFLAGS) -o $(DPCPP_EXE_NAME)


#Parameters: input_filename, gpu(0)/cpu(1), step1, step2, step3, samplingrange, countfilter, batch_size, n_batches

run:
	./embedjoin g
	
	en320ks.txt 1 0 0 0 5000 1 3000 150 0
    
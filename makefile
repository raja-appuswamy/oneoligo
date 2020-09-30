CXX=dpcpp
CXXFLAGS= -O2 -std=c++17 -fsycl -fsycl-unnamed-lambda $(PARAMS) #-gline-tables-only -fdebug-info-for-profiling
LDFLAGS= -lsycl -ltbb -lpthread #-littnotify -ldl
DPCPP_EXE_NAME=embedjoin

build:	#src/embedjoin_dpcpp.o	src/verification.o
	dpcpp $(CXXFLAGS) src/embedjoin_dpcpp.cpp src/verification.cpp src/Time.cpp src/utils.cpp $(LDFLAGS) -o embedjoin


#Parameters: input_filename, gpu(0)/cpu(1)/both(2), samplingrange, countfilter, batch_size, n_batches

# Gen Dataset

run1:
	./embedjoin gen320ks.txt 0  5000 1 10000 45
    
run2:
	./embedjoin gen320ks.txt 1  5000 1 10000 45


run3:
	./embedjoin gen320ks.txt 2 5000 1 10000 45


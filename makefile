PARAMS= -DNUM_STR=7 -DNUM_HASH=16 -DNUM_BITS=12 -DNUM_CHAR=4 -DK_INPUT=150 -DNUM_REP=3

DPCPP=dpcpp
DPCPP_FLAGS= -O2 -std=c++17 -fsycl -fsycl-unnamed-lambda $(PARAMS) 

CLANG=clang++
CLANG_FLAGS= -O2 -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl -fsycl-unnamed-lambda $(PARAMS) 

LDFLAGS= -lsycl -ltbb -lpthread -lboost_program_options
EXE_NAME=onejoin

build:
	$(CLANG) $(CLANG_FLAGS) src/main.cpp src/embedjoin_dpcpp.cpp src/verification.cpp src/Time.cpp src/utils.cpp $(LDFLAGS) -o $(EXE_NAME)

build-dpcpp:
	$(DPCPP) $(DPCPP_FLAGS) src/main.cpp src/embedjoin_dpcpp.cpp src/verification.cpp src/Time.cpp src/utils.cpp $(LDFLAGS) -o $(EXE_NAME)




# Gen Dataset

run1:
	./$(EXE_NAME)  --read gen320ks.txt --device 0 --samplingrange 5000 --countfilter 1 --batch_size 10000 
    
run2:
	./$(EXE_NAME)  --read gen320ks.txt --device 1 --samplingrange 5000 --countfilter 1 --batch_size 3000

run3:
	./$(EXE_NAME)  --read gen320ks.txt --device 2 --samplingrange 5000 --countfilter 1 --batch_size 10000


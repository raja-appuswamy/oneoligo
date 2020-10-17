CXX=dpcpp
PARAMS=-DNUM_STR=7 -DNUM_HASH=16 -DNUM_BITS=12 -DNUM_CHAR=4 -DK_INPUT=150 -DNUM_REP=3
CXXFLAGS= -O2 -std=c++17 -fsycl -fsycl-unnamed-lambda $(PARAMS) 
LDFLAGS= -lsycl -ltbb -lpthread -lboost_program_options
DPCPP_EXE_NAME=onejoin

build:
	$(CXX) $(CXXFLAGS) src/main.cpp src/embedjoin_dpcpp.cpp src/verification.cpp src/Time.cpp src/utils.cpp $(LDFLAGS) -o $(DPCPP_EXE_NAME)




# Gen Dataset

run1:
	./$(DPCPP_EXE_NAME)  --read gen320ks.txt --device 0 --samplingrange 5000 --countfilter 1 --batch_size 10000 
    
run2:
	./$(DPCPP_EXE_NAME)  --read gen320ks.txt --device 1 --samplingrange 5000 --countfilter 1 --batch_size 10000

run3:
	./$(DPCPP_EXE_NAME)  --read gen320ks.txt --device 2 --samplingrange 5000 --countfilter 1 --batch_size 10000


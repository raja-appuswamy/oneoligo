PARAMS=-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50

DPCPP=dpcpp
DPCPP_FLAGS= -O3 -std=c++17 -fsycl -fsycl-unnamed-lambda $(PARAMS) 

CLANG=clang++
CLANG_FLAGS= -O3 -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl -fsycl-unnamed-lambda $(PARAMS) 

LDFLAGS= -lsycl -ltbb -lpthread -lboost_program_options -lboost_thread -lboost_system -lboost_log -lboost_log_setup
EXE_NAME=onejoin

SRC=src
BUILD=build

build:
	$(CLANG) $(CLANG_FLAGS) $(SRC)/main.cpp $(SRC)/embedjoin_dpcpp.cpp $(SRC)/verification.cpp $(SRC)/Time.cpp $(SRC)/utils.cpp $(SRC)/DBSCAN.cpp $(SRC)/constants.cpp $(LDFLAGS) -o $(EXE_NAME)


	
build-dpcpp: $(BUILD)/main.o $(BUILD)/embedjoin_dpcpp.o $(BUILD)/verification.o $(BUILD)/Time.o $(BUILD)/utils.o $(BUILD)/DBSCAN.o $(BUILD)/constants.o
	$(CXX) $(DPCPP_FLAGS) $(BUILD)/main.o $(BUILD)/embedjoin_dpcpp.o $(BUILD)/verification.o $(BUILD)/Time.o $(BUILD)/utils.o $(BUILD)/DBSCAN.o $(BUILD)/constants.o $(LDFLAGS) -o $(EXE_NAME)


update:
	rm $(BUILD)/constants.o && make build-dpcpp

$(BUILD)/constants.o: $(SRC)/constants.cpp
	$(CXX) $(DPCPP_FLAGS) $(PARAMS) $< -c -o $@

$(BUILD)/%.o: $(SRC)/%.cpp
	$(CXX) $(DPCPP_FLAGS)  $< -c -o $@

clean:
	rm $(BUILD)/* 

# Gen Dataset

run1:
	./$(EXE_NAME)  --read gen320ks.txt --device 0 --samplingrange 5000 --countfilter 1 --batch_size 10000 
    
run2:
	./$(EXE_NAME)  --read gen320ks.txt --device 1 --samplingrange 5000 --countfilter 1 --batch_size 3000

run3:
	./$(EXE_NAME)  --read gen320ks.txt --device 2 --samplingrange 5000 --countfilter 1 --batch_size 10000


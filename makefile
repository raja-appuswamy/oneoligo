CXX=dpcpp
PARAMS=-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50
CXXFLAGS= -O3 -std=c++17 -fsycl 
LDFLAGS= -lsycl -ltbb -lpthread -lboost_program_options -lboost_thread -lboost_system -lboost_log -lboost_log_setup
DPCPP_EXE_NAME=onejoin
SRC=src
BUILD=build

build: $(BUILD)/main.o $(BUILD)/embedjoin_dpcpp.o $(BUILD)/verification.o $(BUILD)/Time.o $(BUILD)/utils.o $(BUILD)/DBSCAN.o $(BUILD)/constants.o
	$(CXX) $(CXXFLAGS) $(BUILD)/main.o $(BUILD)/embedjoin_dpcpp.o $(BUILD)/verification.o $(BUILD)/Time.o $(BUILD)/utils.o $(BUILD)/DBSCAN.o $(BUILD)/constants.o $(LDFLAGS) -o $(DPCPP_EXE_NAME)


update:
	rm $(BUILD)/constants.o && make

$(BUILD)/constants.o: $(SRC)/constants.cpp
	$(CXX) $(CXXFLAGS) $(PARAMS) $< -c -o $@

$(BUILD)/%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS)  $< -c -o $@

clean:
	rm $(BUILD)/* 

# Gen Dataset

run1:
	./$(DPCPP_EXE_NAME)  --read gen320ks.txt --device 0 --samplingrange 5000 --countfilter 1 --batch_size 10000 
    
run2:
	./$(DPCPP_EXE_NAME)  --read gen320ks.txt --device 1 --samplingrange 5000 --countfilter 1 --batch_size 10000

run3:
	./$(DPCPP_EXE_NAME)  --read gen320ks.txt --device 2 --samplingrange 5000 --countfilter 1 --batch_size 10000


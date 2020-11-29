.PHONY: build
PARAMS=-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50

DPCPP=dpcpp
DPCPP_FLAGS= -O3 -std=c++17 -fsycl -fsycl-unnamed-lambda $(PARAMS) 

CLANG=clang++
CLANG_FLAGS= -O3 -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl -fsycl-unnamed-lambda $(PARAMS) 

LDFLAGS= -lsycl -ltbb -lpthread -lboost_program_options -lboost_thread -lboost_system -lboost_log -lboost_log_setup
EXE_NAME=onejoin

SRC=src
BUILD=build

build: $(BUILD)/main.o $(BUILD)/onejoin_dpcpp.o $(BUILD)/verification.o $(BUILD)/Time.o $(BUILD)/utils.o $(BUILD)/DBSCAN.o $(BUILD)/constants.o
	$(DPCPP) $(DPCPP_FLAGS) $(BUILD)/main.o $(BUILD)/onejoin_dpcpp.o $(BUILD)/verification.o $(BUILD)/Time.o $(BUILD)/utils.o $(BUILD)/DBSCAN.o $(BUILD)/constants.o $(LDFLAGS) -o $(EXE_NAME)


update:
	rm $(BUILD)/constants.o && make build

$(BUILD)/constants.o: $(SRC)/constants.cpp
	$(DPCPP) $(DPCPP_FLAGS) $(PARAMS) $< -c -o $@

$(BUILD)/%.o: $(SRC)/%.cpp
	$(DPCPP) $(DPCPP_FLAGS)  $< -c -o $@

clean:
	rm $(BUILD)/* 

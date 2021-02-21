.PHONY: build
PARAMS=-DDEF_NUM_STR=5 -DDEF_NUM_HASH=7 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=6 -DDEF_SHIFT=6

DPCPP=dpcpp
DPCPP_FLAGS= -O3 -std=c++17 -fsycl -fsycl-unnamed-lambda -w $(PARAMS) 

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


test-join:
	./onejoin -a 1 -r test/dummy_dataset.txt -d 2 -s 91 -c 0 --batch_size 500

test-cluster:
	./onejoin -a 2 -r test/dummy_dataset.txt -d 2 -s 91 -c 0 --batch_size 500 --min_pts 3



clean:
	rm $(BUILD)/* 

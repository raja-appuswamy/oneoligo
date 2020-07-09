CXX=dpcpp
CXXFLAGS= -O2 $(PARAMS)
LDFLAGS= -fsycl -lsycl -ltbb
DPCPP_EXE_NAME=embedjoin


build:	src/embedjoin_dpcpp.o	src/verification.o
	dpcpp $(CXXFLAGS) src/embedjoin_dpcpp.o src/verification.o $(LDFLAGS) -o embedjoin


#Parameters: input_filename, gpu(0)/cpu(1), step1, step2, step3, samplingrange, countfilter, batch_size, n_batches

run:
	./embedjoin reducedGen320ks.txt 1 0 0 0 5000 1 30000 10
    
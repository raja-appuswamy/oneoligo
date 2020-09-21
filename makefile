CXX=dpcpp
CXXFLAGS= -O2 $(PARAMS)
LDFLAGS= -fsycl -lsycl -ltbb
DPCPP_EXE_NAME=embedjoin


build:	src/embedjoin_dpcpp.o	src/verification.o
	dpcpp $(CXXFLAGS) src/embedjoin_dpcpp.o src/verification.o $(LDFLAGS) -o embedjoin




# Gen Dataset

run1:
	./embedjoin gen320ks.txt 0  5000 1 10000 45
    
run2:
	./embedjoin gen320ks.txt 1  5000 1 10000 45


run3:
	./embedjoin gen320ks.txt 2 5000 1 10000 45


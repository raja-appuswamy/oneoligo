# Oneoligo

### Usage:

#### Cuda-Backend: 

```
mkdir build
make build PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50 -DDEF_CLU_CHUNK_SIZE=10000000"
```

Update parameters:

```
make build PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50 -DDEF_CLU_CHUNK_SIZE=10000000"
```


#### DPCPP:

```
mkdir build
make build-dpcpp PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50"

```
Update parameters:

```
make update PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50"
```

#### Example:

Clustering: 
```
./onejoin --alg 2 --read dataset_name --device 1 --samplingrange 5000 --countfilter 1 --batch_size 10000 --min_pts 2
```

Join:
```
./onejoin --read dataset_name --device 1 --samplingrange 5000 --countfilter 1 --batch_size 10000
```
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
make build-dpcpp PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50 -DDEF_CLU_CHUNK_SIZE=10000000"

```
Update parameters:

```
make update PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50 -DDEF_CLU_CHUNK_SIZE=10000000"
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

#### Constants:

* DEF_NUM_STR: Number of random strings to use for embedding
* DEF_NUM_HASH: Number of hash functions
* DEF_NUM_BITS: Number of hash function bits
* DDEF_NUM_CHAR: Dictionary size
* DDEF_K_INPUT: Edit distance threshold
* DDEF_SHIFT: Shift for embedding
* DDEF_CLU_CHUNK_SIZE: Number of input strings to use in one cluster iteration (affect clustering algorithm only)


#### Program Parameters:
--read: dataset-path <br>
--alg: 1.Join [Default] 2.Cluster <br>
--device: 0.CPU 1.GPU <br>
--samplingrange: max digit to embed <br>
--countfilter: threshold used for filtering candidates <br>
--batch_size: Number of input strigns to embed at time <br><br>
--min_pts: DBSCAN parameter (affect clustering only) <br>
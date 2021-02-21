# Oneoligo Project

### Usage: 

These parameters are just an example. In general, they depend on the dataset used.

```
mkdir build

make build PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50"

./onejoin --read dataset_name --device 0 --samplingrange 5000 --countfilter 1 --batch_size 10000
```


### Update constants:

```
make update PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50"
```


### Simple test:

mkdir build
make -j
make test-join
make test-cluster

### Other links

[Intel article: Enable DNA Storage on Heterogeneous Architectures with oneAPI](https://software.intel.com/content/www/us/en/develop/articles/dna-storage-heterogeneous-architectures-oneapi.html) <br>
[DevMesh project](https://devmesh.intel.com/projects/oneoligo)



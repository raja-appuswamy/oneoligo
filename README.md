# Oneoligo

### Usage:

#### Cuda-Backend: 

```
mkdir build
make build PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50"
```

Update parameters:



#### DPCPP:

```
mkdir build
make build-dpcpp PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50"

```
Update parameters:

```
make update PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50"
```
# Oneoligo

### Usage:

```
mkdir build

make build PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50"

./onejoin --read dataset_name --device 0 --samplingrange 5000 --countfilter 1 --batch_size 10000
```


### Update constants:

```
make update PARAMS="-DDEF_NUM_STR=7 -DDEF_NUM_HASH=16 -DDEF_NUM_BITS=12 -DDEF_NUM_CHAR=4 -DDEF_K_INPUT=150 -DDEF_SHIFT=50"
```
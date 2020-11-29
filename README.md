# oneoligo

### Usage:

```
mkdir build

make build PARAMS="-DNUM_STR=7 -DNUM_HASH=16 -DNUM_BITS=12 -DNUM_CHAR=4 -DK_INPUT=150 -DSHIFT=50"

./onejoin --read dataset_name --device 0 --samplingrange 5000 --countfilter 1 --batch_size 10000
```
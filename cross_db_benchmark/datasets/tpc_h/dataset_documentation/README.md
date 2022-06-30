# Dataset Documentation

Standard TPC-H Dataset

```
git clone git@github.com:electrum/tpch-dbgen.git
make
./dbgen -s 1

for i in `ls *.tbl`; do
    sed 's/|$//' $i > ${i/tbl/csv}
    echo $i;
done
```

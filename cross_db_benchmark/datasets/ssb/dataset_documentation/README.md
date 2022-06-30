# Dataset Documentation

Standard SSB Dataset

In the makefile change to linux

```

git clone https://github.com/gregrahn/ssb-kit.git
cd ssb-kit/dbgen
sed "s/MACHINE =MAC/MACHINE =LINUX/g" makefile -i
sed "s/-O -DDBNAME/-DDBNAME/g" makefile -i
make

rm *.tbl
rm *.csv

# date does not work
SSB_SCALE=2
./dbgen -s $SSB_SCALE -T lineorder
./dbgen -s $SSB_SCALE -T customer
./dbgen -s $SSB_SCALE -T part
./dbgen -s $SSB_SCALE -T supplier
# stupid hack to circumvent IO error
echo "tmp" > date.tbl
./dbgen -s $SSB_SCALE -T date

for i in `ls *.tbl`; do
    sed 's/|$//' $i > ${i/tbl/csv}
    echo $i;
done
```

# Dataset Documentation

Initially, it was taken from a relational fit dataset.

https://relational.fit.cvut.cz/dataset/Airline

However, there were only 400K flights included. Hence, we decided to download more flights from this webpage.

https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time

We extracted the csv files

find . -name 'On_*.csv' -exec cp {} . \;

...and merged the csv files using awk 'FNR > 1' *.csv > merged.csv and then used the script.py in this folder.
#!/bin/bash

dos2unix expected.txt 
sed -i 's/Parsed: \t/[\"/g' expected.txt
sed -i 's/Unigrams: \t/ \", \"/g' expected.txt
sed -i 's/Bigrams: \t/ \", \"/g' expected.txt
sed -i 's/Trigrams: \t/ \", \"/g' expected.txt
tr -d '\n' <expected.txt >test1.txt
rm expected.txt
mv test1.txt expected.txt
echo ' "]' >> expected.txt

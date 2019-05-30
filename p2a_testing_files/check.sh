#!/bin/bash

./cleantext.py test.json > output.txt
cat input.txt > expected.txt
./textedit.sh

diff -u expected.txt output.txt

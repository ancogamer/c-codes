#!/bin/bash
gcc main.c -o app

for file in ./testdata/*.in; do
cat $file | ./app > "${file%.*}"'_my_resout'.out | git diff "${file%.*}".out "${file%.*}"'_my_resout'.out;
done;
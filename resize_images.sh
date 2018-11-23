#!/bin/bash

dir=$1

for i in $(ls ${dir}); do
    convert ${dir}${i} -resize 25% ${dir}${i}
done
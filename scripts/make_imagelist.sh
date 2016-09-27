#!/bin/sh

cwd=`dirname ${0}`
cwd=`cd ${cwd} && pwd`
sl=${cwd}/../dataset/sequencelist.txt

for dd in `find ${cwd}/../dataset -maxdepth 1 -type d | grep 2011`
do
    for d in `find ${dd} -maxdepth 1 -type d | grep extract`
    do
        il=${cwd}/../dataset/imagelist_`basename ${d}`.txt
        find $d | grep png | sort > ${il}
        echo ${il} >> ${sl}
    done
done

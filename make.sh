#!/usr/bin/env bash


usage()
{
    echo "usage: ./make.sh [[[-nc]] | [-h]]"
}

CUDA=
while [ "$1" != "" ]; do
    case $1 in
        -nc | --no-cuda )       CUDA=-nc
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

# Build faster RCNN
echo "[*] Building Faster RCNN"
cd src/faster_rcnn/
./make.sh+$CUDA

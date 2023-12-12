#!/bin/bash

cur_tb=$1
cur_size_limit=$2

sudo find /mnt/d/temp/SyncTablesBetweenClusters/${cur_tb} -type f -size -${cur_size_limit}c -exec rm -f {} \;

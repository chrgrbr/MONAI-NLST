#!/usr/bin/env bash

./build.sh

docker save nlst_monai | gzip -c > NLST_Monai.tar.gz

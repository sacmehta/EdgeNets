#!/usr/bin/env bash
global_path='../../../vision_datasets'
data_dir=$global_path'/pascal_voc'

mkdir -p $data_dir
cd $data_dir

echo "Downloading train and validation images"

wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget -c http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz


echo "unziping files"

tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf benchmark.tgz

echo "Deleting xip and tar files"

rm -rf VOCtrainval_06-Nov-2007.tar
rm -rf VOCtest_06-Nov-2007.tar
rm -rf VOCtrainval_11-May-2012.tar
rm -rf benchmark.tgz

echo "Done"

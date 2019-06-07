#!/usr/bin/env bash
global_path='../../../../vision_datasets'
data_dir=$global_path'/pascal_voc'

mkdir -p $data_dir
cd $data_dir

echo "Downloading data"

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar

rm -rf VOCtrainval_11-May-2012.tar

cd VOCdevkit/VOC2012/

wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip
wget http://cs.jhu.edu/~cxliu/data/list.zip

unzip SegmentationClassAug.zip
unzip SegmentationClassAug_Visualization.zip
unzip list.zip

rm -rf SegmentationClassAug.zip
rm -rf SegmentationClassAug_Visualization.zip
rm -rf list.zip

echo "Done"

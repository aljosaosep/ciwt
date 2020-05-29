#!/bin/bash

# Place your det dirs here
DETDIRS=(/work/aljosa/datasets/kitti_tracking/preproc/training/detection/det_per_seq/det_02 /work/aljosa/datasets/kitti_tracking/preproc/training/detection/det_per_seq/det_02_3DOP /work/aljosa/datasets/kitti_tracking/preproc/training/detection/det_per_seq/det_02_regionlets)

DESTDIR=/work/aljosa/datasets/kitti_tracking/preproc/training/detection/det_per_frame


for DETDIR in ${DETDIRS[@]}; do
	NAME=`basename $DETDIR`
	DIROUT=$DESTDIR/$NAME
	echo $DIROUT
	mkdir -p $DIROUT

	for SEQ in $DETDIR/*; do
		python split_detections_sequence_to_frame.py --sequence_path $SEQ --output_dir $DIROUT
	done
done

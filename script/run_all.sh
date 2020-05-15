#!/bin/bash


SCRIPT_DIR=`dirname "$0"`
#echo ${SCRIPT_DIR}
#exit

DATASET=/work/aljosa/datasets/kitti_tracking/training
PREPROC=/work/aljosa/datasets/kitti_tracking/preproc/training
OUT_DIR=/tmp/tracker_output

# Possible modes: 'detection', 'detection_shape'
# The latter is considered obsolete; this option is still here for eval purposes
MODE=detection # detection_shape

# Determine sequences here
SEQUENCES=`seq 0 20`

# Specify detectors (should correspond to detector dir names and config names)
DETS=(det_02_regionlets det_02_3DOP)

# Path to your binary
CIWT=/usr/wiss/aljosa/projects/remote_sync/cmake-build-debug-remote/apps/CIWTApp

# Exec tracker for each detection set
for DET in ${DETS[@]}; do
	# Exec tracker for each sequence
	for SEQ in ${SEQUENCES[@]}; do
		SEQNAME=`printf "%04d" $SEQ`
		echo "Process: $SEQNAME"

		IM_LEFT=$DATASET/image_02/$SEQNAME/%06d.png
		IM_RIGHT=$DATASET/image_03/$SEQNAME/%06d.png
		DISP=$PREPROC/disparity/disp_$SEQNAME_%06d_left.png

		CALIB=$DATASET/calib/$SEQNAME.txt
		PROP=$PREPROC/cached_proposals/$SEQNAME_%06d.bin
		OUT_PER_DET=${OUT_DIR}/$DET
		DETPATH=$PREPROC/detection/$DET/$SEQNAME/%06d.txt
		DETCFG=${SCRIPT_DIR}/../cfg/$DET.cfg
		$CIWT --config $DETCFG --left_image_path ${IM_LEFT} --right_image_path ${IM_RIGHT} --left_disparity_path ${DISP} --calib_path $CALIB --object_proposal_path $PROP --detections_path $DETPATH --output_dir ${OUT_DIR_DET} --tracking_mode $MODE  --sequence_name $SEQNAME --dataset_name kitti --debug_level 3
	done
done

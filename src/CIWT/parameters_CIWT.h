/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep (osep -at- vision.rwth-aachen.de)

rwth_mot framework is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

rwth_mot framework is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
rwth_mot framework; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#ifndef GOT_PARAMETERS_CIWT_H_H
#define GOT_PARAMETERS_CIWT_H_H

#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace CIWTApp {
    void InitParameters(po::options_description &options) {

        options.add_options()
                ///  --- Object Detections preproc (KITTI). ---
                ("detection_threshold_car", po::value<double>()->default_value(1.0), "Det. threshold: Car")
                ("detection_threshold_pedestrian", po::value<double>()->default_value(3.5), "Det. threshold: Pedestrian")
                ("detection_threshold_cyclist", po::value<double>()->default_value(200), "Det. threshold: Cyclist (100, 2)")
                ("detection_nms_iou", po::value<double>()->default_value(0.6), "Non-maxima-suppression threshold.")
                ("detection_type", po::value<std::string>()->default_value("2D"), "Detection type: 2D or 3D?")

                ///  --- Object Proposals preproc. ---
                ("proposals_min_points", po::value<int>()->default_value(100), "Proposals: min num. points.")
                ("proposals_max_height_offset", po::value<double>()->default_value(2.0), "Proposals: max height offset.")
                ("proposals_min_height", po::value<double>()->default_value(0.5), "Proposals: min. height of the proposal.")

                ///  --- Tracking Etc ---
                ("tracking_e1", po::value<double>()->default_value(1.41), "CRF model params: e1 (min score const.).")
                ("tracking_e3", po::value<double>()->default_value(1.044), "CRF model params: e3 (physical overlap).")
                ("tracking_e4", po::value<double>()->default_value(36.51), "CRF model params: e4 (inlier overlap).")
                ("id_handling_overlap_threshold", po::value<double>()->default_value(0.5), "Overlap thresh for id handling.")
                ("tracking_exp_decay", po::value<int>()->default_value(33), "CRF model params: Exp. decay (unaries).")
                ("max_hole_size", po::value<int>()->default_value(9), "Hypothesis generation: Max. hole size.")
                ("tracking_temporal_window_size", po::value<int>()->default_value(3), "Hypothesis generation: Temporal window size.")
                ("hole_penalty_decay_parameter", po::value<double>()->default_value(1.044), "Hypothesis generation: Hole-penalty parameter.")
                ("min_observations_needed_to_init_hypothesis", po::value<int>()->default_value(2), "Hypothesis generation: Min. obs. to init hypo")
                ("tracking_use_stereo_max_dist", po::value<double>()->default_value(30.0), "Hypothesis generation: Max. dist at which stereo meas. are used.")
                ("tracking_single_detection_hypo_threshold", po::value<double>()->default_value(0.993307149076), "Hypothesis generation: Detections above this thresh start hypo from single det.")

                ///  --- Tracking Area ---
                ("tracking_exit_zones_lateral_distance", po::value<double>()->default_value(0.3), "Exit-zones-dist. for hypothesis termination (lat.)")
                ("tracking_exit_zones_rear_distance", po::value<double>()->default_value(1.3), "Exit-zones-dist. for hypothesis termination (rear)")
                ("tracking_exit_zones_far_distance", po::value<double>()->default_value(80), "Exit-zones-dist. for hypothesis termination (far)")

                ///  --- Data Association ---
                ("association_appearance_model_weight", po::value<double>()->default_value(0.4), "Data association: Appearance model weight.")
                ("association_weight_distance_from_camera_param", po::value<double>()->default_value(0.07), "Data association: Dist-from-camera weight.)")
                ("gaiting_appearance_threshold", po::value<double>()->default_value(0.525), "Data association (gaiting): Appearance model threshold.")
                ("gaiting_IOU_threshold", po::value<double>()->default_value(0.354), "Data association (gaiting): IoU threshold.")
                ("gaiting_mh_distance_threshold", po::value<double>()->default_value(6.52), "Data association (gaiting): Motion model (Mahalanobis dist.) threshold.")
                ("gaiting_min_association_score", po::value<double>()->default_value(0.428), "Data association (gaiting):  Total assoc. score threshold.")
                ("gaiting_size_2D", po::value<double>()->default_value(0.776), "Data association (gaiting):  Bounding-box size.")

                ///  --- Etc ---
                ("max_num_input_observations", po::value<int>()->default_value(500), "Max. observations (per-frame) passed to the tracker.")
                ("debug_mode", po::value<bool>()->default_value(false), "Debug mode on/off.")

                ///  --- Kalman Filter ---
                ("dt", po::value<double>()->default_value(0.1), "Time diff. between two measurements.")
                ("kf_2d_observation_noise", po::value<double>()->default_value(5.0), "KF-2D: Obs. noise")
                ("kf_2d_initial_variance", po::value<double>()->default_value(1000.0), "KF-2D: Initial variance")
                ("kf_2d_system_noise", po::value<double>()->default_value(2.0), "KF-2D: System noise")
                ("kf_init_velocity_variance_car_x", po::value<double>()->default_value(1.0), "KF-3D: initial velocity variance (X).")
                ("kf_init_velocity_variance_car_z", po::value<double>()->default_value(45.0), "KF-3D: initial velocity variance (Z).")
                ("kf_system_pos_variance_car_x", po::value<double>()->default_value(0.5), "KF-3D: pos. sys. variance (X).")
                ("kf_system_pos_variance_car_z", po::value<double>()->default_value(0.5), "KF-3D: pos. sys. variance (Z).")
                ("kf_system_velocity_variance_car_x", po::value<double>()->default_value(1.0), "KF-3D: vel. sys. variance (X).")
                ("kf_system_velocity_variance_car_z", po::value<double>()->default_value(1.0), "KF-3D: vel. sys. variance (X).")

                ///  --- Observation pos. uncertainty prior ---
                ("pos_unc_det_x", po::value<double>()->default_value(0.22), "Detection position unc. prior (X).")
                ("pos_unc_det_z", po::value<double>()->default_value(0.31), "Detection position unc. prior (Z).")
                ("pos_unc_seg_x", po::value<double>()->default_value(0.5), "Segmentation position unc. prior (X).")
                ("pos_unc_seg_z", po::value<double>()->default_value(0.183), "Segmentation position unc. prior (Z).")


                ///  --- Observation Fusion ---
                ("observations_min_association_score", po::value<double>()->default_value(0.00000000001), "Observation fusion: Min. association score.")
                ("observations_e1", po::value<double>()->default_value(0.0005), "Observation fusion: Assoc. score threshold.")
                ("observations_e2", po::value<double>()->default_value(1.0), "Observation fusion: Assoc. soft penalty.")
                ("observations_e3", po::value<double>()->default_value(100), "Observation fusion: Assoc. hard penalty.")
                ("observations_pose_parameter", po::value<double>()->default_value(0.1), "Observation fusion: Pose term weight.")
                ("observations_bbox_parameter", po::value<double>()->default_value(5.0), "Observation fusion: IoU (proj.) term weight.")
                ("observations_size_prior_parameter", po::value<double>()->default_value(0.5), "Observation fusion: Size prior term weight.")
                ("observations_pixel_count_in_parameter", po::value<double>()->default_value(0.1), "Observation fusion: Mask-in-bbox param.")
                ("observations_pixel_count_out_parameter", po::value<double>()->default_value(1.0), "Observation fusion: Mask-out-of-the-bbox param.")
                ("observations_assoc_iou_threshold", po::value<double>()->default_value(0.5), "Observation fusion: assoc. IoU threshold.")
                ("observations_assoc_pose_threshold", po::value<double>()->default_value(3.99), "Observation fusion: Pose (Mahalanobis dist.) threshold.")
                ("observations_assoc_size_threshold", po::value<double>()->default_value(5.66), "Observation fusion: Size (Mahalanobis dist.) threshold.")
                ;
    }
}

#endif //GOT_PARAMETERS_CIWT_H_H

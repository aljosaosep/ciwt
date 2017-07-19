/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Dennis Mitzel (osep, mitzel -at- vision.rwth-aachen.de)

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

#ifndef GOT_UTILS_TRACKING
#define GOT_UTILS_TRACKING

// std
#include <set>

// eigen
#include <Eigen/Core>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

// pcl
#include <pcl/common/common.h>

// utils
#include "sun_utils/camera.h"
#include "sun_utils/utils_kitti.h"

// tracking lib
#include <tracking/hypothesis.h>

// viso, need for EstimateEgomotion
#include <libviso2/viso_stereo.h>

// Forward declarations
namespace GOT { namespace segmentation { class ObjectProposal; } }

namespace GOT {
    namespace tracking {
        namespace utils {
            /**
               * @brief Calls VO module and provides egomotion estimation for the current frame
               */
            Eigen::Matrix4d EstimateEgomotion(libviso2::VisualOdometryStereo &viso, const cv::Mat &color_left, const cv::Mat &color_right);

            /**
               * @brief Given 3D pos., current-frame camera and past-state camera, gives you 'projected' egomotion vector
               */
            Eigen::Vector2d ProjectEgomotion(const Eigen::Vector4d &reference_pose_camera_space, const SUN::utils::Camera &cam_current_frame,
                                             const SUN::utils::Camera &cam_previous_frame);

            /**
               * @brief Transforms 'hypo' instance to 'SUN::kitti::TrackingLabel', ready for evaluation export
               */
            SUN::utils::KITTI::TrackingLabel HypoToLabelDefault(int frame, const GOT::tracking::Hypothesis &hypo, double image_width, double image_height);

            /**
               * @brief Takes hypos and export function, returns labels
               */
            void HypothesisSetToLabels(int frame, const GOT::tracking::HypothesesVector &hypos, std::vector<SUN::utils::KITTI::TrackingLabel> &labels,
                                       std::function<SUN::utils::KITTI::TrackingLabel(int, const GOT::tracking::Hypothesis &)> hypo_to_label_fnc);

            /**
               * @brief Exports observation set to labels, suitable for quantitative evaluation.
               */
            void ObservationSetToLabels(int frame, const std::vector<GOT::tracking::Observation> &observations,
                                        std::vector<SUN::utils::KITTI::TrackingLabel> &labels);

            /**
               * @brief Compute pose via robust analysis of 2D bounding-box content.
               * @author Author: Aljosa (osep@vision.rwth-aachen.de).
               */
            bool ComputeDetectionPoseUsingStereo(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud, const Eigen::Vector4d &bounding_box_2d,
                                                 Eigen::Vector4d &median_footpoint);

            /**
               * @brief Smooth poses by avreaging.
               * @author Author: Dennis Mitzel (mitzel@vision.rwth-aachen.de).
               */
            std::vector<Eigen::Vector4d> SmoothTrajectoryPoses(const std::vector<Eigen::Vector4d> &poses, int kernel_size);
        }
    }
}

#endif

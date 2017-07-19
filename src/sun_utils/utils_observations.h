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

#ifndef GOT_UTILS_OBSERVATIONS_H
#define GOT_UTILS_OBSERVATIONS_H

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

// eigen
#include <Eigen/Core>

// opencv
#include <opencv2/core/core.hpp>

// utils
#include "camera.h"

namespace SUN {
    namespace utils {
        namespace observations {

            /**
             * @brief Computes velocity from the velocity measurements, corresponding to the 3D segment
             * @param[in] velocity_map Assumes velocity encoded in the channels (cv::Mat is of type float). Important: NaN values mean no velocity measurements!
             * @param[in] indices Segmentation mask
             * @param[in] dt Delta-time between two consecutive frames
             * @param[out] Velocity estimate for the given object
             * @author Aljosa (osep@vision.rwth-aachen.de)
             */
            Eigen::Vector3d ComputeVelocity(const cv::Mat &velocity_map,
                                            const std::vector<int> &indices, double dt);

            /**
             * @brief Implements GPB(1), approximation of a Gaussian mixture with an single Gaussian
             *      -> Basically, collapses Mixture-Of-Gaussians into single one.
             * @param[in] pose_3d Estimated 3D position of the object
             * @param[in] indices Object segmentation mask
             * @param[in] P_left Projection matrix of the left camera
             * @param[in] P_right Projection matrix of the right camera
             * @param[in] min_num_points Min. points needed to estimate the covariance
             * @param[out] covariance_matrix_3d Estimated covariance matrix (ret. by ref.)
             * @author Aljosa (osep@vision.rwth-aachen.de)
             */
            bool ComputePoseCovariance(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                       const Eigen::Vector4d &pose_3d,
                                       const std::vector<int> &indices,
                                       const Eigen::Matrix<double, 3, 4> &P_left,
                                       const Eigen::Matrix<double, 3, 4> &P_right,
                                       Eigen::Matrix3d &covariance_matrix_3d, int min_num_points=10);

            /**
             * @brief Compute color histogram, as proposed in W. Choi: Near-Online Multi-Target Tracking with Aggregated Local Flow Descriptor (ICCV'15)
             * @author Aljosa (osep@vision.rwth-aachen.de)
             */
            Eigen::VectorXd ComputeColorHistogramNOMT(const cv::Mat &image, const Eigen::Vector4d &bounding_box_2d, int num_bins_per_channel);

            /**
            * @brief Find a (noisy) ground-plane pose estimate of an detection by casting a ray through lower-midpoint and intersecting it with the ground plane
            * @params camera Current-frame camera pose and intrinsic's.
            * @params bounding_box_2d Detection bounding-box.
            * @params ground_plane The estimated point-cloud ground-plane params [nx ny nx d].
            * @author Author: Aljosa (osep@vision.rwth-aachen.de), adapted from impl. of Dennis Mitzel
            */
            Eigen::Vector4d GetDetectionFootpointFromImageBoundingBox(const SUN::utils::Camera &camera, const Eigen::Vector4d &bounding_box_2d);
        }
    }
}

#endif //GOT_UTILS_OBSERVATIONS_H

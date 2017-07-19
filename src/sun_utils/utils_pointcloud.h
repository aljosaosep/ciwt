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


#ifndef SUN_UTILS_POINTCLOUD_H
#define SUN_UTILS_POINTCLOUD_H

// std
#include <cstdint>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>

namespace SUN { namespace utils { class Camera; } }

namespace SUN {
    namespace utils {
        namespace pointcloud {

            /**
             * @brief Converts disparity-map to pointcloud with specified pose. Assumes pointCloud points to allocated memory!
             * @author Aljosa (osep@vision.rwth-aachen.de), adapted from Francis (engelmann@vision.rwth-aachen.de)
             */
            void ConvertDisparityMapToPointCloud (
                    const cv::Mat& disparity_map,
                    const cv::Mat& color_image,
                    float c_u,
                    float c_v,
                    float focal_len,
                    float baseline,
                    const Eigen::Matrix4d& pose,
                    const bool withNaN,
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud);


            /**
             * @brief Takes raw 360 LiDAR scan, returns portion that intersects with camera frustum (organized RGBD point cloud)
             * @author Aljosa (osep@vision.rwth-aachen.de)
             */
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr RawLiDARCloudToImageAlignedAndOrganized(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr raw_lidar_cloud,
                                                                                            const Eigen::Matrix4d &T_lidar_to_cam,
                                                                                            const cv::Mat &image, const SUN::utils::Camera &camera);
            /**
            * @brief Computes ego-flow map (using visual-odometry estimates) and projected velocity (from scene-flow).
             * This function computes you 'ego flow' map, 'velocity flow' map and 'going out-of-bounds' map.
            *        This are two separate components of optical flow (indeed, optical_flow = ego_flow + velocity_flow).
            *        The third map masks pixels that will leave image area in next frame, flow estimate is thus unreliable.
             * @param[in] scene_cloud Point cloud of the current frame
             * @param[in] flow_cloud Velocity vector cloud (not full scene flow!)
             * @param[in] camera_current_frame Current-frame camera
             * @param[in] camera_past_frame Past-frame camera
             * @param[in] current_frame
             * @param[out] projected_velocity_map Ego-motion component of the optical flow
             * @param[out] projected_ego_map Velocity component of the optical flow
             * @param[out] going_out_of_bounds_map Masks about-to-leave pixels
            */
            void ComputeProjectedSceneAndEgoFlow(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                                 const cv::Mat &velocity_map_3D,
                                                 const SUN::utils::Camera &camera_current_frame, const SUN::utils::Camera &camera_past_frame,
                                                 int current_frame,
                                                 cv::Mat &projected_velocity_map,
                                                 cv::Mat &projected_ego_map,
                                                 cv::Mat &going_out_of_bounds_map);
        }
    }
}

#endif

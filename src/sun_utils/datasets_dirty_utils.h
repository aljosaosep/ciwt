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

#ifndef GOT_DATASETS_DIRTY_UTILS_H
#define GOT_DATASETS_DIRTY_UTILS_H

// OpenCV
#include <opencv2/highgui/highgui.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>

// Boost
#include <boost/program_options.hpp>

// Why fwd dcl. no work here?!?
#include "sun_utils/camera.h"
#include "sun_utils/disparity.h"
#include "detection.h"
#include "utils_kitti.h"

namespace po = boost::program_options;

namespace SUN {
    namespace utils {
        namespace dirty {

            void ComputeDisparityElas(const cv::Mat &image_left, const cv::Mat &image_right,
                                      SUN::DisparityMap &disparity_left, SUN::DisparityMap &disparity_right);

            class DatasetAssitantDirty {

                /// Inputs:
                // 1) Image path
                // 2) Disparity path
                // 3) Depth path
                // 4) VO path
                // 5) Ground-plane path


                /// Outputs
                // Left image (cv::Mat)
                // Disparity (cv::Mat)
                // Depth (cv::Mat)
                // Point-cloud (pcl::PointCloud<pcl::PointXYZRGBA>)
                // Flow-cloud (pcl::PointCloud<pcl::PointXYZNormal>)
                // Left camera, right camera

            public:

                DatasetAssitantDirty(const po::variables_map &config_variables_map);

                bool LoadData__KITTI(int current_frame);

                // bool LoadData__KITTI__DISPARITY(int current_frame);

                // bool LoadData__SCHIPOOL(int current_frame);

                // bool LoadData__RWTH(int current_frame);

                bool LoadData(int current_frame, const std::string dataset_string);

                bool RequestDisparity(int frame, bool save_if_not_avalible=true);


            //private:
                cv::Mat left_image_;
                cv::Mat right_image_;
                SUN::DisparityMap disparity_map_;
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr left_point_cloud_;
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr left_point_cloud_velodyne_;
                cv::Mat velocity_map_;
                SUN::utils::Camera left_camera_;
                SUN::utils::Camera right_camera_;
                double stereo_baseline_;
                //std::vector<SUN::utils::Detection> object_detections_;
                //std::vector<SUN::utils::KITTI::TrackingLabel> kitti_detections_full_sequence_;
                Eigen::Vector4d ground_plane_;
                std::vector<SUN::utils::KITTI::TrackingLabel> parsed_det_;


                po::variables_map variables_map_;
            };
        }
    }
}


#endif //GOT_DATASETS_DIRTY_UTILS_H

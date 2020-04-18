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

#ifndef GOT_UTILS_FILTERING_H
#define GOT_UTILS_FILTERING_H

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

namespace SUN { namespace utils { class GroundModel; } }

// Forward declarations
namespace SUN { namespace utils { class Camera; } }

namespace SUN {
    namespace utils {
        namespace filter {
            /**
               * @brief Keeps only percentage of points, rejects those too far from centroid. Used for bounding box computation.
               * @param[in] input_cloud
               * @param[in] percentage
               * @return Filtered point-cloud.
               * @author Aljosa (osep@vision.rwth-aachen.de)
               */
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr FilterPointCloudBasedOnRadius(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr input_cloud, double percentage);
            pcl::PointIndices FilterPointCloudBasedOnRadius(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud, const std::vector<int> &indices, double percentage);

            /**
               * @brief Filters "bleeding" and outlier points based on (successful) normal estimation.
               * @param[in] cloud_to_be_cleaned
               * @param[in] only_color_outlier_points
               * @author Aljosa (osep@vision.rwth-aachen.de)
               */
            // pcl::PointCloud<pcl::Normal>::Ptr FilterPointCloudBasedOnNormalEstimation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_be_cleaned, bool only_color_outlier_points=false);


            /**
               * @brief Filters points based on distance from ground plane. WARNING: Requires gp-rectified point cloud!
               * @param[in] cloud_to_be_cleaned
               * @param[in] minDistance
               * @param[in] maxDistance
               * @param[in] only_color_outlier_points
               * @return Nothing. Filtering is done directly on input cloud.
               * @author Aljosa (osep@vision.rwth-aachen.de)
               */
            void FilterPointCloudBasedOnDistanceToGroundPlane(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_be_cleaned,
                                                              std::shared_ptr<SUN::utils::GroundModel> ground_model,
                                                              const double minDistance=-0.15, const double maxDistance=3.0, bool only_color_outlier_points=false);
        }
    }
}


#endif //GOT_UTILS_FILTERING_H

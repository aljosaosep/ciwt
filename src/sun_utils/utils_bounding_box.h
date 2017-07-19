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

#ifndef GOT_UTILS_BOUNDING_BOX_H
#define GOT_UTILS_BOUNDING_BOX_H

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

// Forward declarations
namespace SUN { namespace utils { class Camera; } }

namespace SUN {
    namespace utils {
        namespace bbox {

            /**
             * @brief IntersectionOverUnion2d
             * @param[in] rect1
             * @param[in] rect2
             * @return Intersection over union, scalar
             * @author Aljosa (osep@vision.rwth-aachen.de)
             */
            double IntersectionOverUnion2d(const Eigen::Vector4d &rect1, const Eigen::Vector4d &rect2);

            Eigen::Vector4d Intersection2d(const Eigen::Vector4d &rect1, const Eigen::Vector4d &rect2);

            /**
               * @brief Returns 2D bounding box. Parametrized by: [min_x, min_y, width, height]. Works with organized point clouds.
               * @param[in] scene_cloud
               * @param[in] indices, correspoding to points
               * @return A 4d vector, representing bounding box params.
               * @author Aljosa (osep@vision.rwth-aachen.de)
               */
            Eigen::Vector4d BoundingBox2d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                          const std::vector<int> &indices, double percentage);

            Eigen::Vector4d BoundingBox2d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr object_cloud, const SUN::utils::Camera &cam, double percentage);

            /**
            * @brief Returns 3D bounding box. Parametrized by: [center_x, center_y, center_z, width, height, depth, qw, qx, qy, qz]. Works with un-organized point clouds.
            * @param[in] cloud_in
            * @param[in] percentage - percentage of considered points, for robust bounding box estimation
            * @return 3d oriented bounding box (OBB).
            * @author Aljosa (osep@vision.rwth-aachen.de)
            */
            Eigen::VectorXd BoundingBox3d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud_in, double percentage);

            Eigen::VectorXd BoundingBox3d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                          const std::vector<int> &indices, double percentage);

            /**
             * @brief Reparametrize bounding box from center-w-h reprezentation to top_point-w-h
             * @param[in] bounding_box_2d Input bounding box 2d.
             * @param[out] Reparametrized bounding box.
             * @author Aljosa (osep@vision.rwth-aachen.de)
             */
            Eigen::Vector4d ReparametrizeBoundingBoxCenterMidToLeftTopPoint(const Eigen::Vector4d &bounding_box_2d);

            /**
             * @brief Reparametrize bounding box from mid-top-w-h reprezentation to top-left-w-h
             * @param[in] bounding_box_2d Input bounding box 2d.
             * @param[out] Reparametrized bounding box.
             * @author Aljosa (osep@vision.rwth-aachen.de)
             */
            Eigen::Vector4d ReparametrizeBoundingBoxCenterTopToLeftTopPoint(const Eigen::Vector4d &bounding_box_2d);
        }
    }
}



#endif //GOT_UTILS_BOUNDING_BOX_H

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

#ifndef SUN_UTILS_GROUND_MODEL_H
#define SUN_UTILS_GROUND_MODEL_H

// eigen
#include <Eigen/Core>

// pcl
#include <pcl/common/common_headers.h>

namespace SUN {
    namespace utils {

        class GroundModel {
        public:
            virtual double DistanceToGround(const Eigen::Vector3d &p) const=0; // Compute distance of 3d point to ground
            virtual Eigen::Vector3d ProjectPointToGround(const Eigen::Vector3d &p) const=0; // Compute projection of 3d point to ground
            virtual Eigen::Vector3d IntersectRayToGround(const Eigen::Vector3d &ray_origin, const Eigen::Vector3d &ray_direction) const=0; // Interesct a ray to the ground
            virtual Eigen::Vector3d Normal(const Eigen::Vector3d &point_3d) const=0; // Returns normal at the specified 3d (ground) point
            virtual void FitModel(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud_in, double height_threshold)=0;
        };

        class PlanarGroundModel : public GroundModel {
        public:
            // Overloaded methods
            double DistanceToGround(const Eigen::Vector3d &p) const;
            Eigen::Vector3d ProjectPointToGround(const Eigen::Vector3d &p) const;

            /**
              * @brief Intersects a camera ray with a plane (so far, used for ground-plane intersection).
              * @param[in] ray_origin Repr. ray origin vector.
              * @param[in] ray_dir Repr. ray direction (unit) vector.
              * @param[out] A 3d point of intersection.
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            Eigen::Vector3d IntersectRayToGround(const Eigen::Vector3d &ray_origin, const Eigen::Vector3d &ray_direction) const;

            /**
              * @brief Gives you a normal of the ground representation at a given query point.
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            Eigen::Vector3d Normal(const Eigen::Vector3d &point_3d) const; // Returns normal at the specified 3d (ground) point

            /**
              * @brief Specific for PlanarGroundModel: plane parameters [nx ny nz d]
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            const Eigen::Vector4d plane_params() const;
            void set_plane_params(const Eigen::Vector4d &plane_params);

            void FitModel(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud_in, double height_threshold=1.4);

            /**
              * @brief Specific for PlanarGroundModel: fit least-squares plane to the RANSAC inliers
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            Eigen::Vector4d FitPlaneToInliersLeastSquares(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &scene_cloud,
                                                                             const std::vector<int> &indices);

        private:
            Eigen::Vector4d plane_params_;
        };
    }
}

#endif
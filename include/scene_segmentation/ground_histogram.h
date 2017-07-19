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

#ifndef GOT_GROUND_PLANE_HISTOGRAM_H
#define GOT_GROUND_PLANE_HISTOGRAM_H

// PCL
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>

// Eigen
#include <Eigen/Core>
#include <opencv2/core/core.hpp>

// Forward declarations
namespace SUN { namespace utils { class Camera; } }

namespace GOT {
    namespace segmentation {
        /**
        * @brief Represents 2D density histogram of points, projected to the ground-plane.
        * @author Aljosa (osep@vision.rwth-aachen.de)
        */
        class GroundHistogram {
        public:
            GroundHistogram();
            GroundHistogram(double length, double depth, double height, int rows, int cols);

            /**
               * @brief Computes density histogram and corresponding 3d pointcloud index map.
               * @param[in] Ground-plane aligned, possibly preprocessed and cleaned 3D point-cloud.
               */
            void ComputeDensityMap(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr input_cloud, const SUN::utils::Camera &camera, bool distance_compensate=true, const double normalization_factor=1500, const double threshold=0.2);
            void ComputeHeightMaps(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr input_cloud, const SUN::utils::Camera &camera);


            /**
               * @brief Given 3d point, returns corresponding cell coords in the gp-histogram.
               * @return Boolean, indicating whether point falls to the histogram bounds. Coords are returned via ref.
               */
            bool GetBinFrom3dPoint(const SUN::utils::Camera &camera, const Eigen::Vector3d &point_3d, int &row, int &col) const;


            /// Setters / getters
            double rows() const;
            double cols() const;
            const Eigen::MatrixXd& density_map() const;
            const Eigen::MatrixXd& max_height_map() const;
            const Eigen::MatrixXd& min_height_map() const;

            /**
               * @brief Gives you a set of points (repr. by indices) that fall in a bin.
               * @return Boolean, indicating whether point falls to the histogram bounds. Coords are returned via ref.
               */
            const Eigen::Matrix<std::vector<int>, Eigen::Dynamic, Eigen::Dynamic>& point_index_map() const;

        private:
            // Ground histogram density & point list
            Eigen::MatrixXd density_map_;
            Eigen::MatrixXd max_height_map_;
            Eigen::MatrixXd min_height_map_;
            Eigen::Matrix<std::vector<int>, Eigen::Dynamic, Eigen::Dynamic> point_index_map_;

            double length_;
            double depth_;
            double height_;
            int cols_;
            int rows_;
        };

    }
}

#endif

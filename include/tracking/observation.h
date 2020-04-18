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

#ifndef GOT_OBSERVATION_H
#define GOT_OBSERVATION_H

// std
#include <memory>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <src/sun_utils/shared_types.h>



// Forward decl.
namespace SUN { namespace shared_types { enum CategoryTypeKITTI; }}

namespace GOT {
    namespace tracking {
        class Observation {
        public:
            Observation();

            // Getters
            const Eigen::Vector4d& footpoint() const;
            const Eigen::Vector3d& velocity() const;
            const Eigen::Matrix3d& covariance3d() const;
            const Eigen::Vector4d& bounding_box_2d() const;
            const SUN::shared_types::CategoryTypeKITTI detection_category() const;
            const double detection_score() const;
            const Eigen::VectorXd &color_histogram() const;
            const bool detection_avalible() const;
            const bool proposal_3d_avalible() const;
            const Eigen::VectorXd& bounding_box_3d() const;
            const double proposal_3d_score() const;
            const std::vector<int> &pointcloud_indices() const;
            const double orientation() const;
            double score() const;

            //const SUN::shared_types::CompressedMask &compressed_mask() const;

            // Setters
            void set_score(double score);
            void set_detection_score(double score);
            void set_detection_category(SUN::shared_types::CategoryTypeKITTI category);
            void set_footpoint(const Eigen::Vector4d &footpoint);
            void set_velocity(const Eigen::Vector3d &velocity);
            void set_covariance3d(const Eigen::Matrix3d& cov3d);
            void set_bounding_box_2d(const Eigen::Vector4d& bounding_box_2d);
            void set_bounding_box_3d(const Eigen::VectorXd& bounding_box_3d);
            void set_color_histogram(const Eigen::VectorXd &color_histogram);
            void set_detection_avalible(bool detection_avalible);
            void set_proposal_3d_avalible(bool proposal_3d_avalible);
            void set_orientation(double orientation);
            void set_pointcloud_indices(const std::vector<int> &indices, int image_width, int image_height);
            void set_proposal_3d_score(double score);


        protected:
            double detection_score_;
            double proposal_3d_score_;
            double score_;
            double orientation_;

            SUN::shared_types::CategoryTypeKITTI detection_category_;
            Eigen::Vector4d footpoint_;

            Eigen::Matrix3d covariance3d_;
            Eigen::Vector4d bounding_box_2d_;
            Eigen::VectorXd color_histogram_;
            Eigen::VectorXd bounding_box_3d_;
            Eigen::Vector3d velocity_;

            bool detection_avalible_;
            bool proposal_3d_avalible_;

            std::vector<int> cached_indices_;
            //SUN::shared_types::CompressedMask compressed_mask_;
        };

        std::string GetCategoryString(SUN::shared_types::CategoryTypeKITTI type);
    }
}

#endif

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

#include <tracking/observation.h>

namespace GOT {
    namespace tracking {
        Observation::Observation() {
            detection_score_ = std::numeric_limits<double>::quiet_NaN();
            proposal_3d_score_ = std::numeric_limits<double>::quiet_NaN();
            score_ = std::numeric_limits<double>::quiet_NaN();
            orientation_ = std::numeric_limits<double>::quiet_NaN();

            footpoint_.setConstant(std::numeric_limits<double>::quiet_NaN());
            covariance3d_.setConstant(std::numeric_limits<double>::quiet_NaN());
            bounding_box_2d_.setConstant(std::numeric_limits<double>::quiet_NaN());
            color_histogram_.setConstant(std::numeric_limits<double>::quiet_NaN());
            bounding_box_3d_.setConstant(std::numeric_limits<double>::quiet_NaN());
            velocity_.setConstant(std::numeric_limits<double>::quiet_NaN());

            detection_avalible_ = false;
            proposal_3d_avalible_ = false;
        }

        // Getters
        const Eigen::Vector4d& Observation::footpoint() const {
            return footpoint_;
        }

        const Eigen::Vector3d& Observation::velocity() const {
            return velocity_;
        }

        const Eigen::Matrix3d& Observation::covariance3d() const {
            return  covariance3d_;
        }

        const Eigen::Vector4d& Observation::bounding_box_2d() const {
            return bounding_box_2d_;
        }

        const SUN::shared_types::CategoryTypeKITTI  Observation::detection_category() const {
            return detection_category_;
        }

        const double Observation::detection_score() const {
            return detection_score_;
        }

        const Eigen::VectorXd& Observation::color_histogram() const {
            return color_histogram_;
        }

        const bool Observation::detection_avalible() const {
            return detection_avalible_;
        }

        const bool Observation::proposal_3d_avalible() const {
            return proposal_3d_avalible_;
        }

        const Eigen::VectorXd& Observation::bounding_box_3d() const {
            return bounding_box_3d_;
        }

        const double Observation::proposal_3d_score() const {
            return proposal_3d_score_;
        }

        const std::vector<int> &Observation::pointcloud_indices() const {
            return cached_indices_;
            //return compressed_mask_.GetIndices();
        }

        // Setters
        void Observation::set_detection_score(double score) {
            detection_score_ = score;
        }

        void Observation::set_detection_category(SUN::shared_types::CategoryTypeKITTI category) {
            detection_category_ = category;
        }

        void Observation::set_footpoint(const Eigen::Vector4d &footpoint) {
            footpoint_ = footpoint;
        }

        void Observation::set_velocity(const Eigen::Vector3d &velocity) {
            velocity_ = velocity;
        }

        void Observation::set_covariance3d(const Eigen::Matrix3d& cov3d) {
            covariance3d_ = cov3d;
        }

        void Observation::set_bounding_box_2d(const Eigen::Vector4d& bounding_box_2d) {
            bounding_box_2d_ = bounding_box_2d;
        }

        void Observation::set_bounding_box_3d(const Eigen::VectorXd& bounding_box_3d) {
            bounding_box_3d_ = bounding_box_3d;
        }

        void Observation::set_color_histogram(const Eigen::VectorXd &color_histogram) {
            color_histogram_ = color_histogram;
        }

        void Observation::set_detection_avalible(bool detection_avalible) {
            detection_avalible_ = detection_avalible;
        }

        void Observation::set_proposal_3d_avalible(bool proposal_3d_avalible) {
            proposal_3d_avalible_ = proposal_3d_avalible;
        }

        void Observation::set_pointcloud_indices(const std::vector<int> &indices, int image_width, int image_height) {
            //compressed_mask_ = SUN::shared_types::CompressedMask(indices, image_width, image_height);
            cached_indices_ = indices;
        }

        void Observation::set_proposal_3d_score(double score) {
            proposal_3d_score_ = score;
        }

        double Observation::score() const {
            return score_;
        }

        void Observation::set_score(double score) {
            score_ = score;
        }

        std::string GetCategoryString(SUN::shared_types::CategoryTypeKITTI  type) {
            std::string category_str = "NA";
            if (type == SUN::shared_types::CAR)
                category_str = "car";
            else if (type == SUN::shared_types::VAN)
                category_str = "van";
            else if (type == SUN::shared_types::TRUCK)
                category_str = "truck";
            else if (type == SUN::shared_types::PEDESTRIAN)
                category_str = "ped.";
            else if (type == SUN::shared_types::CYCLIST)
                category_str = "cyc.";
            else if (type == SUN::shared_types::UNKNOWN_TYPE)
                category_str = "unk.";

            return category_str;
        }

        void Observation::set_orientation(double orientation) {
            orientation_ = orientation;
        }

        const double Observation::orientation() const {
            return orientation_;
        }

//        const SUN::shared_types::CompressedMask &Observation::compressed_mask() const {
//            return this->compressed_mask_;
//        }
    }
}

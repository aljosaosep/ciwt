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

#include <tracking/hypothesis.h>

namespace GOT {
    namespace tracking {

        Hypothesis::Hypothesis() {
            this->terminated_ = false;
            this->id_ = -1;
            kalman_filter_ = nullptr;
            last_frame_selected_ = -1;
            creation_timestamp_ = -1;
        }

        void Hypothesis::AddInlier(const HypothesisInlier &inlier) {
            this->inliers_.push_back(inlier);
        }

        void Hypothesis::AddEntry(const Eigen::Vector4d &position, int frame_of_detection) {
            this->poses_.push_back(position);
            //this->poses_with_timestamp_.insert(std::pair<int, Eigen::Vector4d>(frame_of_detection, position));
            this->timestamps_.push_back(frame_of_detection);
        }

        // Setters / Getters
        int Hypothesis::id() const {
            return this->id_;
        }

        void Hypothesis::set_id(int id) {
            this->id_ = id;
        }

        bool Hypothesis::terminated() const {
            return terminated_;
        }

        void Hypothesis::set_terminated(bool terminated) {
            terminated_ = terminated;
        }

        const std::vector<Eigen::VectorXd>& Hypothesis::bounding_boxes_3d() const {
            return bounding_boxes_3d_;
        }

        void Hypothesis::set_bounding_boxes_3d(const std::vector<Eigen::VectorXd>& bboxes_3d) {
            bounding_boxes_3d_ = bboxes_3d;
        }

        void Hypothesis::add_bounding_box_3d(const Eigen::VectorXd bbox_3d) {
            this->bounding_boxes_3d_.push_back(bbox_3d);
        }

        const std::vector<Eigen::Vector4d>& Hypothesis::poses() const {
            return poses_;
        }

        void Hypothesis::set_poses(const std::vector<Eigen::Vector4d>& poses) {
            poses_ = poses;
        }

        const std::vector<int>& Hypothesis::timestamps() const {
            return timestamps_;
        }

        void Hypothesis::set_timestamps(const std::vector<int>& timestamps) {
            timestamps_ = timestamps;
        }

        const std::vector<HypothesisInlier> &Hypothesis::inliers() const {
            return inliers_;
        }

        void Hypothesis::set_inliers(const std::vector<HypothesisInlier>& inliers) {
            inliers_ = inliers;
        }

        CoupledExtendedKalmanFilter::Ptr& Hypothesis::kalman_filter() {
            return this->kalman_filter_;
        }

        CoupledExtendedKalmanFilter::ConstPtr Hypothesis::kalman_filter_const() const {
            return this->kalman_filter_;
        }

        const double Hypothesis::score() const {
            return score_;
        }

        void Hypothesis::set_score(double score) {
            score_ = score;
        }

        const Eigen::VectorXd& Hypothesis::color_histogram() const {
            return color_histogram_;
        }

        void Hypothesis::set_color_histogram(const Eigen::VectorXd color_histogram) {
            color_histogram_ = color_histogram;
        }

        Eigen::Vector3d Hypothesis::GetDirectionVector(int max_num_positions) const {
            const int start_offset = std::max(0, static_cast<int>(poses_.size())-max_num_positions);
            Eigen::Vector3d dir;
            dir.setZero();
            int i=0;
            for (i=start_offset; (i+1)<poses_.size(); i+=2) {
                const auto &pos1 = poses_.at(i);
                const auto &pos2 = poses_.at(i+1);
                dir += (pos2-pos1).head(3);
            }

            if (i>0)
                dir /= static_cast<double>(i);

            return dir;
        }

        int Hypothesis::last_frame_selected() const {
            return last_frame_selected_;
        }

        void Hypothesis::set_last_frame_selected(int last_frame_selected) {
            last_frame_selected_ = last_frame_selected;
        }

        void Hypothesis::add_bounding_box_2d_with_timestamp(int timestamp, const Eigen::Vector4d &bbox) {
            this->bounding_boxes_2d_with_timestamp_.insert(std::pair<int, Eigen::Vector4d >(timestamp, bbox));
        }

        void Hypothesis::set_bounding_boxes_2d_with_timestamp(const std::map< int, Eigen::Vector4d > &bboxes) {
            this->bounding_boxes_2d_with_timestamp_ = bboxes;
        }

        const std::map< int, Eigen::Vector4d > &Hypothesis::bounding_boxes_2d_with_timestamp() const {
            return this->bounding_boxes_2d_with_timestamp_;
        }

        const std::vector<Eigen::Vector3d> &Hypothesis::poses_camera_space() const {
            return this->poses_camera_space_;
        }

        void Hypothesis::add_pose_camera_space(const Eigen::Vector3d &pose_cam_space) {
            this->poses_camera_space_.push_back(pose_cam_space);
        }

        void Hypothesis::set_poses_camera_space(const std::vector<Eigen::Vector3d> &poses_cam_space) {
            this->poses_camera_space_ = poses_cam_space;
        }
    }
}

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

#ifndef GOT_HYPOTHESIS_H
#define GOT_HYPOTHESIS_H

// Eigen
#include <Eigen/Core>


// Tracking
#include <tracking/resource_manager.h>
#include <tracking/extended_kalman_filter.h>

// Forward declarations
namespace GOT { namespace tracking { class HypothesisInlier; } }
namespace SUN { namespace utils {  class Camera; } }
namespace SUN { namespace shared_types { enum CategoryTypeKITTI; }}

namespace GOT {
    namespace tracking {

        struct HypothesisInlier {
            int timestamp_; // Timestamp of the inlier
            int index_; // Index of the inlier
            double association_score_; // Inlier-to-hypo 'fitness' score
            double inlier_score_; // Score of the inlier (eg. detection score, proposal score, ...)

            std::vector<double> assoc_data_; // You might wanna store additional assoc. info here for dbg. purposes

            HypothesisInlier(int timestamp, int index, double inlier_score, double association_score) {
                AddInlier(timestamp, index, inlier_score, association_score);
            }

            void AddInlier(int timestamp, int index, double inlier_score, double association_score) {
                this->timestamp_ = timestamp;
                this->index_ = index;
                this->inlier_score_ = inlier_score;
                this->association_score_ = association_score;
            }
        };

//        namespace bayes_filter {
//            SUN::shared_types::CategoryTypeKITTI GetCategoryType(const std::vector<double> &posterior, std::function<SUN::shared_types::CategoryTypeKITTI(int)> index_to_category_f);
//            std::vector<double> CategoryFilter(const std::vector<double> &likelihood,
//                                               const std::vector<double> &prior);
//        }


        /**
           * @brief Represents a basic tracker unit.
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class Hypothesis {
        public:

            Hypothesis();

            /**
             * @brief Adds hypo pose+timestamp pair. Note: pose in world-space.
             */
            void AddEntry(const Eigen::Vector4d &position, int frame_of_detection);

            /**
             * @brief Adds inlier info to the hypothesis. Inlier is uniquely determined by frame number and detection table index.
             */
            void AddInlier(const HypothesisInlier &inlier);

            /**
             * @brief Computes direction vector from last max_num_positions. If hypo size is smaller that max_num_position,
             * it takes only available positions.
             * @param max_num_positions Specifies number of past positions used for computing direction vector.
             */
            Eigen::Vector3d GetDirectionVector(int max_num_positions=4) const;

            // Setters / Getter
            int id() const;
            void set_id(int id);

            bool terminated() const;
            void set_terminated(bool terminated);

            const double score() const;
            const Eigen::VectorXd& color_histogram() const;
            const std::vector<Eigen::VectorXd>& bounding_boxes_3d() const;
            void set_bounding_boxes_3d(const std::vector<Eigen::VectorXd>& bboxes_3d);
            void add_bounding_box_3d(const Eigen::VectorXd bbox_3d);

            void set_poses_camera_space(const std::vector<Eigen::Vector3d> &poses_cam_space);
            void add_pose_camera_space(const Eigen::Vector3d &pose_cam_space);
            const std::vector<Eigen::Vector3d>& poses_camera_space() const;

            const std::vector<Eigen::Vector4d>& poses() const;
            void set_poses(const std::vector<Eigen::Vector4d>& poses);

            const std::map<int, Eigen::Vector4d>& bounding_boxes_2d_with_timestamp() const;
            void set_bounding_boxes_2d_with_timestamp(const std::map<int, Eigen::Vector4d> &bboxes);
            void add_bounding_box_2d_with_timestamp(int timestamp, const Eigen::Vector4d &bbox);

            const std::vector<int>& timestamps() const;
            void set_timestamps(const std::vector<int>& timestamps);

            const std::vector<HypothesisInlier>& inliers() const;
            void set_inliers(const std::vector<HypothesisInlier>& inliers);

            void set_score(double score);
            void set_color_histogram(const Eigen::VectorXd color_histogram);

            CoupledExtendedKalmanFilter::Ptr& kalman_filter();
            CoupledExtendedKalmanFilter::ConstPtr kalman_filter_const() const;

            int last_frame_selected() const;
            void set_last_frame_selected(int last_frame_selected);

            std::vector<double> category_probability_distribution_;
            int creation_timestamp_;

        protected:
            // Cached values
            std::vector<Eigen::Vector4d> poses_;
            std::vector<Eigen::Vector3d> poses_camera_space_; // Need for export!
            std::vector<int> timestamps_;
            std::vector<HypothesisInlier> inliers_;
            std::vector<Eigen::VectorXd> bounding_boxes_3d_;
            std::map<int,  Eigen::Vector4d > bounding_boxes_2d_with_timestamp_;
            CoupledExtendedKalmanFilter::Ptr kalman_filter_;
            Eigen::VectorXd color_histogram_;

            int id_;
            bool terminated_;
            double score_;
            int last_frame_selected_;
        };

        typedef std::vector<Hypothesis> HypothesesVector;
    }
}


#endif

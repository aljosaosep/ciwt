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

#include "detection.h"

// utils
#include "shared_types.h"
#include "camera.h"
#include "ground_model.h"
#include "utils_observations.h"
#include "utils_bounding_box.h"

namespace SUN {
    namespace utils {
        namespace detection {

            std::vector<Detection> NonMaximaSuppression(const std::vector<Detection> &detections_in, double iou_threshold) {
                std::vector<Detection> set_being_processed = detections_in;
                std::vector<Detection> set_suppressed;

                while (set_being_processed.size()>0) {
                    auto it_to_max_element = std::max_element(set_being_processed.begin(), set_being_processed.end(), [](const Detection &d1, const Detection &d2){ return d1.score() < d2.score();  });
                    auto best_scoring_det = *it_to_max_element;

                    std::vector<Detection> overlapping_set;
                    std::set<int> overlapping_set_inds;

                    // Find overlapping detections
                    for (int i=0; i<set_being_processed.size(); i++) {
                        const auto &element_to_test = set_being_processed.at(i);
                        double IOU = SUN::utils::bbox::IntersectionOverUnion2d(best_scoring_det.bounding_box_2d(), element_to_test.bounding_box_2d());
                        if (IOU>iou_threshold) {
                            overlapping_set.push_back(element_to_test);
                            overlapping_set_inds.insert(i);
                        }
                    }

                    // Filter!
                    set_suppressed.push_back(best_scoring_det);
                    std::vector<Detection> remaining_det;
                    for (int i=0; i<set_being_processed.size(); i++) {
                        if (overlapping_set_inds.count(i)==0 && overlapping_set_inds.size()>0)  {
                            remaining_det.push_back(set_being_processed.at(i));
                        }
                    }
                    set_being_processed = remaining_det;
                }

                return set_suppressed;
            }

            std::vector<Detection> GeometricFilter(const std::vector<Detection> &detections_in, const SUN::utils::Camera &camera,
                                                   double apparent_size_max,
                                                   double apparent_size_min,
                                                   double max_width_percentage) {
                std::vector<Detection> detections_fine;
                for (const auto &det : detections_in) {
                    // NOPE to geometrically-unfeasible detections
                    const Eigen::Vector4d &det_bbox_2d = det.bounding_box_2d();
                    const Eigen::Vector4d &detection_footpoint = det.footpoint();
                    const double apparent_size = (static_cast<double>(det_bbox_2d[3]) * detection_footpoint[2]) / camera.f_u();
                    if (apparent_size > apparent_size_max) // Detected objects should not be > 2m. (plus one 'slack' m)
                        continue;
                    if ((apparent_size < apparent_size_min) && (detection_footpoint[2] < 30)) // Detected objects should not be too small (check only in 30m range)
                        continue;
                    if (det_bbox_2d[2] > static_cast<double>(camera.width()) * max_width_percentage) // det. width > half_image_width?
                        continue;
                    detections_fine.push_back(det);
                }
                return detections_fine;
            }

            std::vector<Detection> ScoreFilter(const std::vector<Detection> &detections_in, std::function<bool(const Detection &detection)> f_score_filt) {
                std::vector<Detection> detections_fine;
                for (const auto &det : detections_in) {
                    if (f_score_filt(det))
                        detections_fine.push_back(det);
                }
                return detections_fine;
            }

            std::vector<Detection> ProjectTo3D(const std::vector<Detection> &detections_in,
                                               const SUN::utils::Camera &left_camera,
                                               const SUN::utils::Camera &right_camera) {
                std::vector<Detection> detections = detections_in;
                for (auto &det : detections) {
                    Eigen::Vector4d detection_footpoint = observations::GetDetectionFootpointFromImageBoundingBox(left_camera, det.bounding_box_2d());
                    detection_footpoint.head<3>() = left_camera.ground_model()->ProjectPointToGround(detection_footpoint.head<3>());

                    Eigen::Matrix3d detection_pose_covariance;
                    SUN::utils::Camera::ComputeMeasurementCovariance3d(detection_footpoint.head<3>(), 0.5,
                                                                       left_camera.P().block(0,0,3,4), right_camera.P().block(0,0,3,4),
                                                                       detection_pose_covariance);
                    det.set_footpoint(detection_footpoint);
                    det.set_pose_covariance_matrix(detection_pose_covariance);
                }

                return detections;
            }
        }

        Detection::Detection() {

        }

        Detection::Detection(const Eigen::Vector4d &bounding_box_2d, int detection_id) {
            this->bounding_box_2d_ = bounding_box_2d;
            this->id_ = detection_id;
        }

        const Eigen::Vector4d &Detection::bounding_box_2d() const {
            return bounding_box_2d_;
        }


        const Eigen::Vector4d &Detection::footpoint() const {
            return footpoint_;
        }

        void Detection::set_footpoint(const Eigen::Vector4d &footpoint) {
            footpoint_ = footpoint;
        }

        void Detection::set_score(double score) {
            score_ = score;
        }

        void Detection::set_pose_covariance_matrix(const Eigen::Matrix3d &pose_covariance_matrix) {
            pose_covariance_matrix_ = pose_covariance_matrix;
        }

        void Detection::set_category(int detection_category) {
            category_ = detection_category;
        }

        void Detection::set_bounding_box_2d(const Eigen::Vector4d &bounding_box_2d) {
            bounding_box_2d_ = bounding_box_2d;
        }

        const Eigen::Matrix3d &Detection::pose_covariance_matrix() const {
            return pose_covariance_matrix_;
        }

        int Detection::category() const {
            return category_;
        }

        double Detection::score() const {
            return score_;
        }

        double Detection::observation_angle() const {
            return observation_angle_;
        }

        void Detection::set_observation_angle(double observation_angle) {
            observation_angle_ = observation_angle;
        }

        const Eigen::VectorXd &Detection::bounding_box_3d() const {
            return bounding_box_3d_;
        }

        void Detection::set_bounding_box_3d(const Eigen::VectorXd &bounding_box_3d) {
            this->bounding_box_3d_ = bounding_box_3d;
        }
    }
}

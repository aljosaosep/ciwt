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

#include "CIWT_tracker.h"

// std
#include <unordered_map>

// tracking lib
#include <tracking/utils_tracking.h>

// utils
#include "sun_utils/ground_model.h"

namespace GOT {
    namespace tracking {
        namespace ciwt_tracker {

            void CIWTTracker::AdvanceHypo(Resources::ConstPtr detections, int reference_frame, bool is_forward, Hypothesis &ref_hypo, bool allow_association) {
                bool detections_lookup_success = false;
                bool camera_lookup_success = false;
                bool camera_prev_lookup_success = false;

                const auto &ref_camera = detections->GetCamera(reference_frame, camera_lookup_success);
                assert(camera_lookup_success);
                const auto &ref_camera_minus_one = detections->GetCamera(reference_frame - 1, camera_prev_lookup_success);
                assert(camera_prev_lookup_success);
                const std::vector<Observation> &ref_state_observations = detections->GetObservations(reference_frame, detections_lookup_success);

                if (detections_lookup_success && camera_lookup_success) {

                    // Get 'projected egomotion' vector
                    Eigen::VectorXd u = Eigen::VectorXd::Zero(15);

                    if (camera_prev_lookup_success) {
                        Eigen::Vector2d pose_gp = ref_hypo.kalman_filter_const()->GetPoseGroundPlane();
                        Eigen::Vector4d plane_cam = std::static_pointer_cast<SUN::utils::PlanarGroundModel>(
                                ref_camera.ground_model())->plane_params();
                        const double height = ref_camera.ComputeGroundPlaneHeightInWorldSpace(pose_gp[0], pose_gp[1], plane_cam);
                        Eigen::Vector4d predicted_pose = Eigen::Vector4d(pose_gp[0], height, pose_gp[1], 1.0);
                        Eigen::Vector4d predicted_pose_cam_space = ref_camera.WorldToCamera(predicted_pose);
                        predicted_pose_cam_space.head<3>() = ref_camera.ground_model()->ProjectPointToGround(
                                predicted_pose_cam_space.head<3>()); // 'snap' to the ground
                        Eigen::Vector2d proj_ego = GOT::tracking::utils::ProjectEgomotion(
                                predicted_pose_cam_space, ref_camera, ref_camera_minus_one);


                        double multip = is_forward ? 1.0 : -1.0;
                        u[7] = multip * proj_ego[0]; // Important to negate - the time arrow is 'inverse'
                        u[8] = multip * proj_ego[1];
                    }


                    this->dynamics_handler_->ApplyTransition(ref_camera, u, ref_hypo);

                    /// Data association
                    std::vector<double> data_association_scores;
                    std::vector<std::vector<double> > data_association_scores_debug_data;
                    bool any_inlier_found = false;
                    if (allow_association)
                        any_inlier_found = AssociateObservationToHypothesis(ref_state_observations, ref_camera, ref_hypo,
                                                                            data_association_scores, data_association_scores_debug_data);
                    int inlier_index = -1;
                    double association_score = 0.0;
                    if (any_inlier_found) {
                        auto max_element_it = std::max_element(data_association_scores.begin(), data_association_scores.end());
                        inlier_index = static_cast<int>(std::distance(data_association_scores.begin(), max_element_it));
                        association_score = data_association_scores.at(inlier_index);
                        const auto &nn_inlier = ref_state_observations.at(inlier_index);
                        auto new_inlier = HypothesisInlier(reference_frame, inlier_index, nn_inlier.score(), association_score);
                        new_inlier.assoc_data_ = data_association_scores_debug_data.at(inlier_index);
                        ref_hypo.AddInlier(new_inlier);
                    }

                    HypothesisUpdate(detections, is_forward, inlier_index, reference_frame, ref_hypo);
                }
            }

            std::vector<Hypothesis> CIWTTracker::ExtendHypotheses(const std::vector<Hypothesis> &current_hypotheses_set, Resources::ConstPtr detections, int current_frame/*, std::vector<bool> &extend_indicator*/) {

                // Max. no. frames for hypothesis extrapolation
                assert(parameter_map_.count("max_hole_size"));
                int max_hole_size = parameter_map_.at("max_hole_size").as<int>();

                std::vector<Hypothesis> new_hypothesis_set;

                detection_indices_used_for_extensions_.clear();

                /// For each existing hypothesis, check, if it can be extended.
                for (int j = 0; j < current_hypotheses_set.size(); j++) {

                    auto &hypo = current_hypotheses_set.at(j);
                    bool allow_association = true;

                    // Stop extending hypo if last inlier is 'too old'
                    if (hypo.timestamps().size() > 0) {
                        // We have seen detection already. See how big the "hole" is.
                        int last_inlier_frame = hypo.inliers().back().timestamp_;
                        int hole_size = std::abs(current_frame - last_inlier_frame);
                        if (hole_size > max_hole_size) {
                            allow_association = false;
                        }
                    }

                    // Do not perform data assoc. if hypo was terminated
                    if (hypo.terminated())
                        allow_association = false;

                    Hypothesis hypo_being_processed = hypo;
                    this->AdvanceHypo(detections, current_frame, true, hypo_being_processed, allow_association);
                    new_hypothesis_set.push_back(hypo_being_processed);
                }

                return new_hypothesis_set;
            }

            std::vector<Hypothesis> CIWTTracker::StartNewHypotheses(Resources::ConstPtr detections, int current_frame) {

                /// Parameters
                double e1 = parameter_map_.at("tracking_e1").as<double>();
                double conf_det_thresh = parameter_map_.at("tracking_single_detection_hypo_threshold").as<double>();
                int temporal_window_size = parameter_map_.at("tracking_temporal_window_size").as<int>();
                int min_obs_init_hypo = parameter_map_.at("min_observations_needed_to_init_hypothesis").as<int>();
                int max_hole_size = parameter_map_.at("max_hole_size").as<int>();
                bool debug_mode = parameter_map_.at("debug_mode").as<bool>();

                std::vector<Hypothesis> new_hypotheses;

                /// Get resources
                bool detections_lookup_success;
                const std::vector<Observation> &observations_current_frame = detections->GetObservations(current_frame, detections_lookup_success);
                const int num_detections_current_frame = observations_current_frame.size();

                if (!detections_lookup_success)
                    return new_hypotheses; // Return empty hypothesis set

                /// Try to init a new hypothesis from each detection
                for (int i = 0; i < num_detections_current_frame; i++) {
                    const auto &observation_current_frame = observations_current_frame.at(i);

                    // If first detection has a high score, let's start hypo directly!
                    double seed_detection_score = observation_current_frame.score();
                    bool is_super_confident_seed_detection = seed_detection_score > conf_det_thresh;
                    int accept_new_hypothesis = true;

                    /// Start new hypothesis
                    Hypothesis new_hypo;
                    new_hypo.set_last_frame_selected(current_frame);
                    double init_score = is_super_confident_seed_detection ? (e1 + 0.1) : 1.0;
                    new_hypo.AddInlier(HypothesisInlier(current_frame, i, observation_current_frame.score(), init_score));
                    HypothesisInit(detections, false, i, current_frame, new_hypo);

                    /// Walk thorough the past detections and find inliers. This is a 'backward walk' (temporal domain).
                    int num_past_frames_for_hypo_construction = std::min(current_frame, temporal_window_size);

                    for (int j = 1; j <= (num_past_frames_for_hypo_construction - 1); j++) {
                        const int past_frame_currently_being_processed = current_frame - j;

                        // Require certain number of supporting detections in first n-frames.
                        if (j <= min_obs_init_hypo) {
                            if ((new_hypo.inliers().size() != j) && !is_super_confident_seed_detection) {
                                accept_new_hypothesis = false;
                                break;
                            }
                        }

                        // Stop extending hypo, if we haven't been able to make association for too long.
                        if (new_hypo.timestamps().size() > 0) {
                            // We have seen detection already. See how big the "hole" is.
                            int last_inlier_frame = new_hypo.inliers().back().timestamp_;
                            int hole_size = std::abs(last_inlier_frame - past_frame_currently_being_processed);

                            if (hole_size > max_hole_size) {
                                break;
                            }
                        }

                        this->AdvanceHypo(detections, past_frame_currently_being_processed, false, new_hypo);
                    }

                    if (accept_new_hypothesis) {
                        if (debug_mode)
                            new_hypo.set_id(this->last_hypo_id_++);
                        new_hypo.creation_timestamp_ = current_frame;
                        new_hypotheses.push_back(new_hypo);
                    }
                }

                /// Post-processing
                // Re-start hypo, flip the inliers
                // Re-filter from the 'first' observation to the current-frame observation

                // Implementation of this part is weird and often seems its not correct ... but it is!
                for (Hypothesis &hypo : new_hypotheses) {

                    std::vector<HypothesisInlier> inliers = hypo.inliers();
                    std::vector<HypothesisInlier> inliers_reverse = hypo.inliers();

                    std::vector<int> timestamps;
                    std::vector<Eigen::Vector4d> poses;
                    std::map< int, Eigen::Vector4d > bboxes_with_timestamp;

                    if (inliers.size() >= 1) {

                        // Inliers
                        std::reverse(inliers.begin(), inliers.end());
                        //hypo.set_inliers(inliers);

                        // Re-set timestamps, poses and bounding-boxes
                        hypo.set_timestamps(timestamps);
                        hypo.set_poses(poses);
                        hypo.set_bounding_boxes_2d_with_timestamp(bboxes_with_timestamp);
                        hypo.set_inliers(std::vector<HypothesisInlier>());
                        hypo.set_poses_camera_space(std::vector<Eigen::Vector3d>());
                        hypo.set_bounding_boxes_3d(std::vector<Eigen::VectorXd>());

                        // Re-start the kalman filter
                        bool first_frame_camera_lookup_success = false, first_frame_detections_lookup_success = false;

                        // Access first inlier: frame, idx, camera, observations
                        const int first_frame = inliers.front().timestamp_;
                        const int first_frame_inlier_index =  inliers.front().index_;
                        const auto &camera_first_frame = detections->GetCamera(first_frame, first_frame_camera_lookup_success);
                        assert(camera_lookup_success);
                        const std::vector<Observation> &observations_2d_first_frame = detections->GetObservations(first_frame, first_frame_detections_lookup_success);

                        if (first_frame_camera_lookup_success && first_frame_detections_lookup_success) {
                            const auto first_frame_obs = observations_2d_first_frame.at(first_frame_inlier_index);
                            hypo.kalman_filter().reset();

                            // Re-set the first inlier
                            auto new_inlier = HypothesisInlier(first_frame, first_frame_inlier_index, first_frame_obs.detection_score(), 1.0);
                            hypo.AddInlier(new_inlier);

                            // Re-init the hypo
                            HypothesisInit(detections, true, first_frame_inlier_index, first_frame, hypo);

                            const int next_frame = inliers.front().timestamp_ +1;
                            const int last_inlier_frame = inliers.back().timestamp_;

                            /// Now loop from first inlier frame to the last inlier frame.
                            int current_inlier_index = 1; // This is sort of pointer to the 'next inlier'
                            for (int candidate_frame = next_frame; candidate_frame <= last_inlier_frame; candidate_frame++) {

                                bool current_frame_camera_lookup_success = false, current_frame_detections_lookup_success = false;

                                // Access the next-inlier-in-the-vec data
                                const int inlier_frame = inliers.at(current_inlier_index).timestamp_;
                                const int inlier_index = inliers.at(current_inlier_index).index_;

                                // Looks weird, but it does what it should
                                const double inlier_assoc_score = inliers_reverse.at(current_inlier_index).association_score_;

                                const auto &camera_candidate_frame = detections->GetCamera(inlier_frame, current_frame_camera_lookup_success);

                                assert(current_frame_camera_lookup_success);

                                const auto &camera_candidate_frame_minus_one = detections->GetCamera(inlier_frame-1, current_frame_camera_lookup_success);
                                assert(current_frame_camera_lookup_success);

                                const std::vector<Observation> &observations_candidate_frame = detections->GetObservations(inlier_frame, current_frame_detections_lookup_success);
                                const auto &obs_current_frame = observations_candidate_frame.at(inlier_index);

                                // Egomotion compensation
                                Eigen::Vector2d pose_gp = hypo.kalman_filter_const()->GetPoseGroundPlane();
                                Eigen::Vector4d plane_cam =  std::static_pointer_cast<SUN::utils::PlanarGroundModel>(camera_candidate_frame.ground_model())->plane_params();
                                const double height = camera_candidate_frame.ComputeGroundPlaneHeightInWorldSpace(pose_gp[0], pose_gp[1], plane_cam);
                                Eigen::Vector4d predicted_pose = Eigen::Vector4d(pose_gp[0], height, pose_gp[1], 1.0);
                                Eigen::Vector4d predicted_pose_cam_space = camera_candidate_frame.WorldToCamera(predicted_pose);
                                predicted_pose_cam_space.head<3>() = camera_candidate_frame.ground_model()->ProjectPointToGround(predicted_pose_cam_space.head<3>()); // 'snap' to the ground
                                Eigen::Vector2d proj_ego = GOT::tracking::utils::ProjectEgomotion(predicted_pose_cam_space, camera_candidate_frame, camera_candidate_frame_minus_one);
                                Eigen::VectorXd u = Eigen::VectorXd::Zero(15);
                                u[7] = proj_ego[0]; // Time arrow is 'forward' again
                                u[8] = proj_ego[1];

                                this->dynamics_handler_->ApplyTransition(camera_candidate_frame, u, hypo);

                                // Only update the state if we have inlier in this frame!
                                if (candidate_frame == inlier_frame) {
                                    auto new_inlier = HypothesisInlier(candidate_frame, inlier_index, obs_current_frame.detection_score(), /* obs_score */ inlier_assoc_score);
                                    hypo.AddInlier(new_inlier);

                                    // Update using inlier
                                    HypothesisUpdate(detections, true, inlier_index, candidate_frame, hypo);
                                    current_inlier_index++; // Inc 'inlier pointer' to the next
                                }
                                else {
                                    // No inlier, extrapolate
                                    HypothesisUpdate(detections, true, -1, candidate_frame, hypo);
                                }
                            }
                            assert(current_inlier_index == inliers.size());
                        }
                        else {
                            std::cout << "StartNewHypotheses::ERROR: can't reach resources!" << std::endl;
                        }
                    }
                }

                return new_hypotheses;
            }
        }
    }
}
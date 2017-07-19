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

// cv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std
#include <unordered_map>

// tracking
#include <tracking/qpbo.h>
#include <tracking/utils_tracking.h>
#include <tracking/category_filter.h>

// utils
#include "utils_visualization.h"
#include "ground_model.h"
#include "utils_bounding_box.h"

// etc
#include "observation_fusion.h"

// CIWT
#include "potential_functions.h"
#include "CIWT_tracker.h"

namespace GOT {
    namespace tracking {
        namespace ciwt_tracker {
            typedef std::function<double(const Hypothesis &hypo, int current_frame)> UnaryFunc;
            typedef std::function<double(const Hypothesis &hypo_1, const Hypothesis &hypo_2, int current_frame)> PairwiseFunc;

            auto CategoryToLikelihoodFnc = [](SUN::shared_types::CategoryTypeKITTI observed_categ)->std::vector<double> {
                const int num_categories = 8; // Car, ped, cyclist
                const double prob_categ = 0.7;
                const double prob_other = (1.0-prob_categ) / (num_categories-1);
                std::vector<double> meas_likelihood(num_categories, prob_other);
                int categ_idx = static_cast<int>(observed_categ);
                meas_likelihood.at(categ_idx) = prob_categ;
                return meas_likelihood;
            };

            CIWTTracker::CIWTTracker(const po::variables_map &params) : MultiObjectTracker3DBase(params) {
                auto dyn_handler_ptr = new  DynamicsHandlerCIWT(params); //dt);
                dynamics_handler_ = std::unique_ptr<GOT::tracking::DynamicsModelHandler>(dyn_handler_ptr);
                //std::unique_ptr<DynamicsHandlerCVTR>(new DynamicsHandlerCVTR(dt));
            }

            Eigen::VectorXi RunInference(const std::vector<Hypothesis> &hypos, int current_frame, UnaryFunc unary_pot, PairwiseFunc pairwise_pot, const po::variables_map &parameter_map) {

                /// Init matrix
                const int num_hypos = (int)hypos.size();
                Eigen::MatrixXd Q(num_hypos, num_hypos);
                Q.setZero();

                /// Compute unaries
                for (int i = 0; i < num_hypos; i++) {
                    const auto &hypo = hypos.at(i);
                    Q(i, i) = unary_pot(hypo, current_frame);
                }

                /// Compute pairwise potentials -> physical overlap penalty
                for (int i = 0; i < num_hypos; i++) {
                    for (int j = i + 1; j < num_hypos; j++) {

                        if (i != j) {
                            // Compute physical overlap penalty
                            const auto &hypo_1 = hypos.at(i);
                            const auto &hypo_2 = hypos.at(j);
                            double overlap_penalty = pairwise_pot(hypo_1, hypo_2, current_frame);

                            if (overlap_penalty > 0.001) {
                                Q(i, j) += -0.5*overlap_penalty;
                                Q(j, i) = Q(i, j);
                            }
                        }
                    }
                }

                /// Run the solver
                Eigen::VectorXi m;
                QPBO::SolveMultiBranch(Q, m); // Result is the binary vector m, indicating 'selected' or 'not selected'

                return m;
            }

            void CIWTTracker::ProcessFrame(Resources::ConstPtr detections, int current_frame) {

                /// Parameters
                double max_dist_for_stereo = parameter_map_.at("tracking_use_stereo_max_dist").as<double>();
                int max_hole_size = parameter_map_.at("max_hole_size").as<int>();
                int nr_frames_not_selected_tolerance = 10;
                bool debug_mode = parameter_map_.at("debug_mode").as<bool>();
                bool perform_model_selection = true;

                /// Access camera from past frame to current
                bool lookup_success = false;
                auto camera = detections->GetCamera(current_frame, lookup_success);

                /// Extend existing hypotheses
                std::vector<Hypothesis> extended_hypos;
                std::vector<bool> hypos_extend_indicator;
                if (hypotheses_.size() > 0)
                    extended_hypos = this->ExtendHypotheses(hypotheses_, detections, current_frame/*, hypos_extend_indicator*/);

                /// From detections, that were not used for extending hypotheses, start new hypotheses
                std::vector<Hypothesis> new_hypos = this->StartNewHypotheses(detections, current_frame);

                printf ("Ext. hypos: %d, new hypos: %d\r\n", (int)extended_hypos.size(), (int)new_hypos.size());

                /// {active_hypos_set} = {extended_set} U {new_set}
                this->hypotheses_.clear();
                this->hypotheses_.insert(hypotheses_.begin(), extended_hypos.begin(), extended_hypos.end());
                this->hypotheses_.insert(hypotheses_.end(), new_hypos.begin(), new_hypos.end());

                /// Terminate hypos that were not supported by detections for too long
                for (auto &hypo:hypotheses_) {
                    const int last_detection_frame = hypo.inliers().back().timestamp_;
                    if (std::abs(current_frame - last_detection_frame) > max_hole_size) {
                        if (this->verbose_)
                            std::cout << "Hypo " << hypo.id() << " received no detections for " << max_hole_size << " frames, terminated." << std::endl << std::endl;
                        hypo.set_terminated(true);
                    }
                }

                /// Terminate hypos that go out of bounds
                this->CheckExitZones(camera, current_frame);

                /// Inference
                Eigen::VectorXi m;
                auto unary_handle = std::bind(CRF::CIWT::hypothesis_selection::potential_func::HypoUnary, std::placeholders::_1, std::placeholders::_2, this->parameter_map_);

                if (perform_model_selection) {
                    //auto unary_handle = std::bind(CRF::CIWT::hypothesis_selection::potential_func::HypoUnary, std::placeholders::_1, std::placeholders::_2, this->parameter_map_);
                    auto pairwise_handle = std::bind(CRF::CIWT::hypothesis_selection::potential_func::HypoPairwise, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this->parameter_map_);
                    m = RunInference(hypotheses_, current_frame, unary_handle, pairwise_handle, this->parameter_map_);
                }
                else {
                    m = Eigen::VectorXi::Ones(hypotheses_.size());
                }

                /// Compute scores (=unaries)
                for (auto &hypo:hypotheses_) {
                    hypo.set_score(unary_handle(hypo, current_frame));
                }

                /// Update ID's (unless we are in the debug mode)
                if (!debug_mode) {
                    this->AssignIDs (m, detections, current_frame, hypotheses_);
                }

                /// Set MDL scores, last_selected
                this->SetLastFrameSelected(hypotheses_, m, current_frame);

                /// Fill the 'selected set'
                selected_set_.clear();
                terminated_hypotheses_.clear();
                for (int i = 0; i < m.size(); i++) {
                    int is_hypo_selected = m[i];
                    if (is_hypo_selected && !(hypotheses_.at(i).terminated())) {
                        selected_set_.push_back(hypotheses_.at(i));
                    }
                    else if (is_hypo_selected && hypotheses_.at(i).terminated()) {
                        terminated_hypotheses_.push_back(hypotheses_.at(i));
                    }
                    else {
                        not_selected_hypos_.push_back(hypotheses_.at(i));
                    }
                }

                /// Remove hypos that weren't selected for a while.
                hypotheses_ = this->RemoveInactiveHypotheses(hypotheses_, current_frame, nr_frames_not_selected_tolerance);
            }

            void CIWTTracker::HypothesisUpdate(Resources::ConstPtr detections, bool is_forward_update, int inlier_index,
                                                         int current_frame,
                                                         Hypothesis &hypo) {


                assert(parameter_map_.count("tracking_use_stereo_max_dist"));
                double max_dist_for_stereo = parameter_map_.at("tracking_use_stereo_max_dist").as<double>();
                //auto bbox_reparam_fnc = SUN::utils::bbox::ReparametrizeBoundingBoxCenterTopToLeftTopPoint;
                auto bbox_reparam_fnc = SUN::utils::bbox::ReparametrizeBoundingBoxCenterMidToLeftTopPoint;

                SUN::utils::Camera camera_curr;
                SUN::utils::Camera camera_prev;
                bool got_cam_1 = detections->GetCamera(current_frame, camera_curr);
                bool got_cam_2 = detections->GetCamera(current_frame, camera_prev);
                assert(got_cam_1 && got_cam_2);


                if (inlier_index > -1 ) {

                    /// => Got inlier, update hypothesis state using this inlier

                    /// Access inlier info
                    Observation nn_observation;
                    bool got_inlier = detections->GetInlierObservation(current_frame, inlier_index, nn_observation);
                    assert(got_inlier);
                    const double observation_dist_to_cam = nn_observation.footpoint()[2];
                    Eigen::Vector4d inlier_footpoint_world = camera_curr.CameraToWorld(nn_observation.footpoint());
                    bool use_velocity_measurement = nn_observation.proposal_3d_avalible() && observation_dist_to_cam<max_dist_for_stereo;

                    /// Filter dynamics
                    this->dynamics_handler_->ApplyCorrection(camera_curr, nn_observation, is_forward_update, hypo);

                    /// Filter category
                    auto category_likelihood = CategoryToLikelihoodFnc(nn_observation.detection_category());
                    GOT::tracking::bayes_filter::CategoryFilter(category_likelihood, hypo.category_probability_distribution_);

                    /// Get filtered pose, size and bounding-box
                    const Eigen::Vector2d &posterior_pose_gp = hypo.kalman_filter_const()->GetPoseGroundPlane();
                    const Eigen::Vector3d &posterior_size_3d = hypo.kalman_filter_const()->GetSize3d();
                    const Eigen::Vector4d kf_bbox2d = hypo.kalman_filter_const()->GetBoundingBox2d();
                    Eigen::Vector4d plane_cam = std::static_pointer_cast<SUN::utils::PlanarGroundModel>(camera_curr.ground_model())->plane_params();
                    double x = posterior_pose_gp[0];
                    double z = posterior_pose_gp[1];
                    double height_world = camera_curr.ComputeGroundPlaneHeightInWorldSpace(x, z, plane_cam);
                    Eigen::Vector4d hypo_position_camera_space = camera_curr.WorldToCamera(Eigen::Vector4d(x, height_world, z, 1.0));

                    hypo.AddEntry(Eigen::Vector4d(posterior_pose_gp[0], height_world, posterior_pose_gp[1], 1.0), current_frame);

                    /// Cache 3D bounding-box, 2D bounding box, pose (world&camera space), color histogram
                    double height_estim = posterior_size_3d[1];
                    Eigen::Vector3d bbox_3d_center_in_camera_space = hypo_position_camera_space.head<3>() + Eigen::Vector3d(0.0, 1.0, 0.0)*(-height_estim/2.0);
                    Eigen::VectorXd filtered_bbox_3d;
                    filtered_bbox_3d.setZero(6);
                    filtered_bbox_3d << bbox_3d_center_in_camera_space[0],
                            bbox_3d_center_in_camera_space[1],
                            bbox_3d_center_in_camera_space[2],
                            posterior_size_3d[0], posterior_size_3d[1], posterior_size_3d[2];

                    hypo.add_bounding_box_3d(filtered_bbox_3d);
                    hypo.add_pose_camera_space(hypo_position_camera_space.head<3>());
                    hypo.add_bounding_box_2d_with_timestamp(current_frame, bbox_reparam_fnc(kf_bbox2d));
                    hypo.set_color_histogram(0.4 * hypo.color_histogram() + 0.6 * nn_observation.color_histogram());
                }

                else {

                    /// => No inlier, extrapolate the state.

                    /// Access state prio (predictions)
                    const Eigen::Vector2d &prediction_pose_gp = hypo.kalman_filter_const()->GetPoseGroundPlane();
                    const Eigen::Vector3d &prediction_size_3d = hypo.kalman_filter_const()->GetSize3d();
                    const Eigen::Vector4d predicted_bounding_box_2d = hypo.kalman_filter_const()->GetBoundingBox2d();
                    Eigen::Vector4d plane_cam =  std::static_pointer_cast<SUN::utils::PlanarGroundModel>(camera_curr.ground_model())->plane_params();
                    double x = prediction_pose_gp[0];
                    double z = prediction_pose_gp[1];
                    double height_world = camera_curr.ComputeGroundPlaneHeightInWorldSpace(x, z, plane_cam);
                    Eigen::Vector4d hypo_position_camera_space = camera_curr.WorldToCamera(Eigen::Vector4d(x, height_world, z, 1.0));
                    Eigen::Vector4d predicted_pose = Eigen::Vector4d(prediction_pose_gp[0], height_world, prediction_pose_gp[1], 1.0);

                    hypo.AddEntry(predicted_pose, current_frame);

                    /// Cache using predictions: 3D bounding-box, 2D bounding box, pose (world&camera space), color histogram
                    double height_estim = prediction_size_3d[1];
                    Eigen::Vector3d bbox_3d_center_in_camera_space = hypo_position_camera_space.head<3>() + Eigen::Vector3d(0.0, 1.0, 0.0)*(-height_estim/2.0);
                    Eigen::VectorXd filtered_bbox_3d;
                    filtered_bbox_3d.setZero(6);
                    filtered_bbox_3d << bbox_3d_center_in_camera_space[0], bbox_3d_center_in_camera_space[1], bbox_3d_center_in_camera_space[2],
                            prediction_size_3d[0], prediction_size_3d[1], prediction_size_3d[2];
                    hypo.add_bounding_box_2d_with_timestamp(current_frame, bbox_reparam_fnc(predicted_bounding_box_2d));
                    hypo.add_pose_camera_space(hypo_position_camera_space.head<3>());
                    hypo.add_bounding_box_3d(filtered_bbox_3d);
                }
            }



            void CIWTTracker::HypothesisInit(Resources::ConstPtr detections, bool is_forward_update, int inlier_index, int current_frame,
                                                       Hypothesis &hypo) {

                //auto bbox_reparam_fnc = SUN::utils::bbox::ReparametrizeBoundingBoxCenterTopToLeftTopPoint;
                auto bbox_reparam_fnc = SUN::utils::bbox::ReparametrizeBoundingBoxCenterMidToLeftTopPoint;

                Observation observation;
                SUN::utils::Camera camera;
                bool got_inlier = detections->GetInlierObservation(current_frame, inlier_index, observation);
                bool got_cam = detections->GetCamera(current_frame, camera);
                assert(got_inlier && got_cam);

                Eigen::Vector4d observation_footpoint_world_space = camera.CameraToWorld(observation.footpoint());
                hypo.AddEntry(observation_footpoint_world_space, current_frame);

                /// Initialize dynamics handler
                this->dynamics_handler_->InitializeState(camera, observation, is_forward_update, hypo);

                /// Initialize category filter
                auto category_likelihood = CategoryToLikelihoodFnc(observation.detection_category());
                hypo.category_probability_distribution_ = std::vector<double>(category_likelihood.size(), 1.0/category_likelihood.size()); // Init the prior
                hypo.category_probability_distribution_ = GOT::tracking::bayes_filter::CategoryFilter(category_likelihood, hypo.category_probability_distribution_);

                /// Init the cached stuff
                Eigen::Vector4d posterior_bounding_box_2d = hypo.kalman_filter_const()->GetBoundingBox2d();
                Eigen::Vector3d posterior_size_3d = hypo.kalman_filter_const()->GetSize3d();
                Eigen::Vector2d posterior_pose_gp = hypo.kalman_filter_const()->GetPoseGroundPlane();

                Eigen::Vector4d plane_cam =  std::static_pointer_cast<SUN::utils::PlanarGroundModel>(camera.ground_model())->plane_params();
                double x = posterior_pose_gp[0];
                double z = posterior_pose_gp[1];
                double height_world = camera.ComputeGroundPlaneHeightInWorldSpace(x, z, plane_cam);
                Eigen::Vector4d hypo_position_camera_space = camera.WorldToCamera(Eigen::Vector4d(x, height_world, z, 1.0));

                double height_estim = posterior_size_3d[1];
                Eigen::Vector3d bbox_3d_center_in_camera_space = hypo_position_camera_space.head<3>() + Eigen::Vector3d(0.0, 1.0, 0.0)*(-height_estim/2.0);

                // Get filtered 3D bounding-box
                Eigen::VectorXd filtered_bbox_3d;
                filtered_bbox_3d.setZero(6);
                filtered_bbox_3d << bbox_3d_center_in_camera_space[0], bbox_3d_center_in_camera_space[1], bbox_3d_center_in_camera_space[2],
                        posterior_size_3d[0], posterior_size_3d[1], posterior_size_3d[2];
                hypo.add_pose_camera_space(hypo_position_camera_space.head<3>());
                hypo.add_bounding_box_2d_with_timestamp(current_frame, bbox_reparam_fnc(posterior_bounding_box_2d));
                hypo.set_color_histogram(observation.color_histogram());
                hypo.add_bounding_box_3d(filtered_bbox_3d);
            }

            auto ProjectCovarianceToGroundPlane = [](const Eigen::Matrix3d &pose_covariance_3d) -> Eigen::Matrix2d {
                Eigen::Matrix2d pose_covariance_2d = Eigen::Matrix2d::Identity();
                pose_covariance_2d(0, 0) = pose_covariance_3d(0, 0);
                pose_covariance_2d(0, 1) = pose_covariance_3d(0, 2);
                pose_covariance_2d(1, 0) = pose_covariance_3d(2, 0);
                pose_covariance_2d(1, 1) = pose_covariance_3d(2, 2);

                return pose_covariance_2d;
            };

            bool  CIWTTracker::AssociateObservationToHypothesis(const std::vector<Observation> &observations,
                                                                          const SUN::utils::Camera &camera,
                                                                          const Hypothesis &hypo,
                                                                          std::vector<double> &observations_association_scores,
                                                                          std::vector< std::vector<double> > &association_scores_debug) {

                //auto bbox_reparam_fnc = SUN::utils::bbox::ReparametrizeBoundingBoxCenterTopToLeftTopPoint; //
                auto bbox_reparam_fnc = SUN::utils::bbox::ReparametrizeBoundingBoxCenterMidToLeftTopPoint;

                association_scores_debug.clear();
                association_scores_debug.resize(observations.size());

                observations_association_scores.clear();
                observations_association_scores.resize(observations.size());
                observations_association_scores.assign(observations.size(), 0.0);
                bool at_least_one_inlier_found = false;

                // Filtered 2D pose on the ground plane
                Eigen::Vector2d kalman_prediction_ground_plane = hypo.kalman_filter_const()->GetPoseGroundPlane();
                Eigen::Matrix2d pred_cov_2d = hypo.kalman_filter_const()->GetPoseCovariance();

                // Filtered image-plane bounding-box
                Eigen::Vector4d bbox_2d_kf = hypo.kalman_filter_const()->GetBoundingBox2d();
                const Eigen::Vector4d &hypo_bbox_2d = bbox_reparam_fnc(bbox_2d_kf);

                // Filtered 3D bounding-box size
                const Eigen::Vector3d &hypo_bbox_3d_size = hypo.kalman_filter_const()->GetSize3d();
                const Eigen::Matrix3d &hypo_bbox_3d_cov = hypo.kalman_filter_const()->GetSizeCovariance();

                const Eigen::VectorXd hypo_color_hist = hypo.color_histogram();


                /// Go through all new observation and find the best fit
                for (int i = 0; i < observations.size(); i++) {
                    const auto &observation = observations.at(i);

                    Eigen::Vector4d obs_bbox_2d = observation.bounding_box_2d();
                    Eigen::VectorXd obs_bbox_3d = observation.bounding_box_3d();
                    Eigen::Vector3d obs_bbox_3d_size = obs_bbox_3d.block(3, 0, 3, 1);
                    Eigen::Vector4d obs_pose_world = camera.CameraToWorld(observation.footpoint());
                    const Eigen::VectorXd obs_color_hist = observation.color_histogram();
                    Eigen::Vector2d obs_pose_gp = Eigen::Vector2d(obs_pose_world[0], obs_pose_world[2]);
                    Eigen::Matrix3d obs_cov_world = camera.R() * observation.covariance3d() * camera.R().transpose();
                    Eigen::Matrix2d obs_cov_gp = ProjectCovarianceToGroundPlane(obs_cov_world);

                    bool invertible;
                    double determinant_size = 0.0;

                    // Compute Mahalanobis distance between the observation and prediction
                    double determinant_pose = 0.0;
                    const Eigen::Vector2d pose_diff = obs_pose_gp - kalman_prediction_ground_plane;
                    Eigen::Matrix2d cov_2d_inverse;

                    (obs_cov_gp).computeInverseAndDetWithCheck(cov_2d_inverse, determinant_pose, invertible);
                    assert(invertible);
                    const double mahalanobis_dist_squared = pose_diff.transpose() * cov_2d_inverse * pose_diff;

                    // Compute IoU (image plane)
                    double IOU_2d = SUN::utils::bbox::IntersectionOverUnion2d(hypo_bbox_2d, obs_bbox_2d);

                    // CHI2INV_2_95 = 5.991464 Inverse of the chi2-cdf with 2 dofs at 0.95.
                    // CHI2INV_2_99 = 9.210340 Inverse of the chi2-cdf with 2 dofs at 0.99.

                    //! Compute weights
                    double denom = std::sqrt(39.47841 * std::abs(determinant_pose));
                    assert(denom > 0.0);
                    double motion_model_weight =  (1.0 / denom) * std::exp(-0.5 * mahalanobis_dist_squared);

                    assert(obs_color_hist.size()==hypo_color_hist.size());
                    double color_intersect_kernel = 0.0;
                    for (int j=0; j<obs_color_hist.size(); j++) {
                        color_intersect_kernel += std::min(obs_color_hist[j], hypo_color_hist[j]);
                    }

                    // --------- FETCH THE PARAMETERS ---------
                    double gaiting_pose_thresh = parameter_map_.at("gaiting_mh_distance_threshold").as<double>();
                    double gaiting_iou_thresh = parameter_map_.at("gaiting_IOU_threshold").as<double>();
                    double gaiting_appearance_thresh = parameter_map_.at("gaiting_appearance_threshold").as<double>();
                    double gaiting_min_association_score = parameter_map_.at("gaiting_min_association_score").as<double>();
                    double gaiting_size_2D = parameter_map_.at("gaiting_size_2D").as<double>();
                    double assoc_apperance_model_weight = parameter_map_.at("association_appearance_model_weight").as<double>();
                    double assoc_weight_distance_from_camera = parameter_map_.at("association_weight_distance_from_camera_param").as<double>();
                    // ------------------------------------

                    double bbox_iou_model_weight = IOU_2d;
                    double hypo_dist_from_camera = hypo.poses_camera_space().back()[2];

                    if ((std::sqrt(mahalanobis_dist_squared) < gaiting_pose_thresh) &&
                            (IOU_2d > gaiting_iou_thresh) &&
                            (color_intersect_kernel > gaiting_appearance_thresh)) { // Gaiting

                        // 2D bounding box size gaiting
                        auto h = hypo_bbox_2d[3];
                        auto h_det = obs_bbox_2d[3];
                        auto w = hypo_bbox_2d[2];
                        auto w_det = obs_bbox_2d[2];
                        double score_size_2D = 1.0 - (std::fabs(h-h_det)/(2*(h+h_det))) - (std::fabs(w-w_det)/(2*(w+w_det)));

                        if (score_size_2D < gaiting_size_2D) // Warning: hard-coded!
                            continue;

                        double w1 = std::exp(-assoc_weight_distance_from_camera*hypo_dist_from_camera);
                        double w2 = 1.0 - w1;
                        //double association_score = w1 * motion_model_weight + w2 * bbox_iou_model_weight;

                        // Incorp. app. model
                        double other_terms_weight = 1.0-assoc_apperance_model_weight;
                        double association_score = other_terms_weight*(w1 * motion_model_weight + w2 * bbox_iou_model_weight) +
                                                   assoc_apperance_model_weight * color_intersect_kernel;

                        if (association_score > gaiting_min_association_score) {
                            at_least_one_inlier_found = true;
                            observations_association_scores.at(i) = association_score;

                            association_scores_debug.at(i).push_back(w1);
                            association_scores_debug.at(i).push_back(w2);
                            association_scores_debug.at(i).push_back(motion_model_weight);
                            association_scores_debug.at(i).push_back(bbox_iou_model_weight);
                            association_scores_debug.at(i).push_back(association_score);
                        }
                    }
                }
                return at_least_one_inlier_found;
            }
        }
    }
}

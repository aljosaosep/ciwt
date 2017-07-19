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

#include "observation_fusion.h"

// utils
#include "sun_utils/ground_model.h"
#include "sun_utils/utils_bounding_box.h"
#include "sun_utils/utils_observations.h"
#include "sun_utils/detection.h"
#include "sun_utils/utils_common.h"

// tracking lib -> need QPBO solver
#include <tracking/qpbo.h>

// pcl
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>

namespace GOT {
    namespace tracking {

        namespace observation_processing {

            /**
             * @brief Computes Mahalanobis distance between learned mean category size and 3D-proposal size (bounding-box-3D).
             * @param[in] bounding_box_3d The 3D bounding box of the proposal
             * @param[in] category The detection category
             * @return Mahalanobis distance squared.
             * @author Aljosa (osep@vision.rwth-aachen.de)
             */
            double DistanceSize3D (const Eigen::VectorXd &bounding_box_3d, SUN::shared_types::CategoryTypeKITTI category) {

                // Measurement
                Eigen::Vector3d measurement_bbox_3d = bounding_box_3d.segment<3>(3);
                Eigen::Matrix3d system_size_variance = Eigen::Matrix3d::Identity();
                system_size_variance.diagonal() = Eigen::Vector3d(proposals_variance_size_estimates[0],
                                                                  proposals_variance_size_estimates[1],
                                                                  proposals_variance_size_estimates[2]);

                // Pick the correct prior
                Eigen::Vector3d prior_bbox_3d;
                Eigen::Matrix3d prior_size_variance = Eigen::Matrix3d::Identity();

                if (category==SUN::shared_types::CAR) {
                    const auto *car_mean = category_size_stats::car_mean_whl;
                    const auto *car_variance = category_size_stats::car_variance_whl;
                    prior_bbox_3d = Eigen::Vector3d(car_mean[0], car_mean[1], car_mean[2]);
                    prior_size_variance.diagonal() = Eigen::Vector3d(car_variance[0], car_variance[1], car_variance[2]);
                }
                else if (category==SUN::shared_types::PEDESTRIAN) {
                    const auto *ped_mean = category_size_stats::ped_mean_whl;
                    const auto *ped_variance = category_size_stats::ped_variance_whl;
                    prior_bbox_3d = Eigen::Vector3d(ped_mean[0], ped_mean[1], ped_mean[2]);
                    prior_size_variance.diagonal() = Eigen::Vector3d(ped_variance[0], ped_variance[1], ped_variance[2]);
                }
                else if (category==SUN::shared_types::CYCLIST) {
                    const auto *cyc_mean = category_size_stats::cyclist_mean_whl;
                    const auto *cyc_variance = category_size_stats::cyclist_variance_whl;
                    prior_bbox_3d = Eigen::Vector3d(cyc_mean[0], cyc_mean[1], cyc_mean[2]);
                    prior_size_variance.diagonal() = Eigen::Vector3d(cyc_variance[0], cyc_variance[1], cyc_variance[2]);
                }

                // Compute Mahalanobis dist.
                Eigen::Vector3d diff = measurement_bbox_3d-prior_bbox_3d;
                double mh_dist_squared = diff.transpose() * (prior_size_variance+system_size_variance).inverse() * diff;
                return mh_dist_squared;
            }

            double UnaryFuncLinearCombination(const std::vector<double> &potentials, const po::variables_map &parameter_map) {
                auto pose_param = parameter_map.at("observations_pose_parameter").as<double>();
                auto bbox_param = parameter_map.at("observations_bbox_parameter").as<double>();
                auto size_param = parameter_map.at("observations_size_prior_parameter").as<double>();
                auto pos_thresh = parameter_map.at("observations_assoc_pose_threshold").as<double>();
                auto size_thresh = parameter_map.at("observations_assoc_size_threshold").as<double>();
                auto iou_thresh = parameter_map.at("observations_assoc_iou_threshold").as<double>();
                auto pixel_in_param = parameter_map.at("observations_pixel_count_in_parameter").as<double>();
                auto pixel_out_param = parameter_map.at("observations_pixel_count_out_parameter").as<double>();

                assert(potentials.size() == 5);
                const double mh_pose_squared = potentials[0];
                const double mh_size_squared = potentials[1];
                const double IoU = potentials[2];
                const double norm_pixel_count_in = potentials[3];
                const double norm_pixel_count_out = potentials[4];

                double assoc_score = -1.0;
                const double footpoint_term = std::exp(-0.5 * mh_pose_squared);
                const double prior_size_term = std::exp(-0.5 * mh_size_squared);

                // Gaiting
                if ((IoU > iou_thresh) && (std::sqrt(mh_pose_squared) < pos_thresh) && (std::sqrt(mh_size_squared) < size_thresh)) {

                    // Compute observation association score
                    assoc_score = pose_param * footpoint_term + bbox_param * IoU + size_param * prior_size_term
                                  + (pixel_in_param * norm_pixel_count_in - pixel_out_param * norm_pixel_count_out);
                }
                return assoc_score;
            }

            double ComputeAssociationScore(const GOT::segmentation::ObjectProposal &prop,
                                           const Eigen::Vector3d &detection_footpoint, const Eigen::Matrix2d detecton_cov2d, const Eigen::Vector4d &detection_bounding_box_2d,
                                           const SUN::shared_types::CategoryTypeKITTI detection_type,
                                           const SUN::utils::Camera &camera,
                                           const po::variables_map &parameter_map) {

                // Proposal pose 3D
                Eigen::Vector4d proposal_footpoint = prop.pos3d();
                proposal_footpoint.head<3>() = camera.ground_model()->ProjectPointToGround(proposal_footpoint.head<3>());

                // Proposal bounding-box 2D
                const Eigen::Vector4d &proposal_bbox_2d = prop.bounding_box_2d();
                const Eigen::Matrix3d &proposal_cov3d = prop.pose_covariance_matrix();

                // Proposal pose-covariance
                Eigen::Matrix2d proposal_cov2d = Eigen::Matrix2d::Identity();
                proposal_cov2d(0, 0) = proposal_cov3d(0, 0);
                proposal_cov2d(0, 1) = proposal_cov3d(0, 2);
                proposal_cov2d(1, 0) = proposal_cov3d(2, 0);
                proposal_cov2d(1, 1) = proposal_cov3d(2, 2);

                // Proposal distance from camera
                double proposal_dist_from_camera = prop.pos3d()[2];

                // Compute pose diff between proposal and footpoint pose on the ground plane
                const Eigen::Vector2d pose_diff(detection_footpoint[0] - proposal_footpoint[0], detection_footpoint[2] - proposal_footpoint[2]);
                bool invertible;
                double determinant;
                Eigen::Matrix2d cov2d_inverse;
                (detecton_cov2d + proposal_cov2d).computeInverseAndDetWithCheck(cov2d_inverse, determinant, invertible);
                if (!invertible) {
                    std::cout << std::endl;
                    std::cout << "Observation processing: Cov. matrix is not invertible! This is serious, fix needed!"<< std::endl;
                    std::cout << std::endl;
                }

                // Compute Mahalanobis distance (pose on ground plane)
                const double mahalanobis_dist_pose_squared = pose_diff.transpose() * cov2d_inverse * pose_diff;

                // Compute Mahalanobis distance: 3D size
                const double mahalanobis_dist_proposal_and_prior_size = DistanceSize3D(prop.bounding_box_3d(), detection_type);

                // Compute IoU (image domain)
                double IOU_2d = SUN::utils::bbox::IntersectionOverUnion2d(detection_bounding_box_2d, proposal_bbox_2d);

                // Compute normalized count of 3d pts inside/outside bbox
                int count_in = 0;
                int count_out = 0;
                for (auto idx:prop.pointcloud_indices()) {
                    int x,y;
                    SUN::utils::UnravelIndex(idx, camera.width(), &x, &y);

                    int x1 = static_cast<int>(detection_bounding_box_2d[0]);
                    int y1 = static_cast<int>(detection_bounding_box_2d[1]);
                    int x2 = static_cast<int>(detection_bounding_box_2d[0] + detection_bounding_box_2d[2]);
                    int y2 = static_cast<int>(detection_bounding_box_2d[1] + detection_bounding_box_2d[3]);

                    if (x>x1 && x<x2 && y>y1 && y<y2)
                        count_in ++;
                    else
                        count_out ++;
                }

                assert(detection_bounding_box_2d[2]*detection_bounding_box_2d[3] > 0.0);

                const double count_in_normalized = static_cast<double>(count_in) / (detection_bounding_box_2d[2]*detection_bounding_box_2d[3]);
                const double count_out_normalized = static_cast<double>(count_out) / (detection_bounding_box_2d[2]*detection_bounding_box_2d[3]);

                std::vector<double> potentials = {mahalanobis_dist_pose_squared, mahalanobis_dist_proposal_and_prior_size,
                                                  IOU_2d, count_in_normalized, count_out_normalized};

                return UnaryFuncLinearCombination(potentials, parameter_map); //obs_assoc_fnc(potentials);
            }


            /**
             * @brief Computes 'mean' 3D bounding box of the object, given it's 3D pose and category.
             * @param[in] category
             * @param[in] footpoint
             * @return Eigen::VectorXd, containing 3D bounding-box [pos_x, pos_y, pos_z, w, h, l]
             * @author Aljosa (osep@vision.rwth-aachen.de)
             */
            Eigen::VectorXd LabelToBoundingBox3D(SUN::shared_types::CategoryTypeKITTI category, const Eigen::Vector4d &footpoint) {

                Eigen::VectorXd bounding_box_3d;
                bounding_box_3d.setZero(6);
                bounding_box_3d.head<3>() = footpoint.head<3>();

                // Use category prior
                if (category == SUN::shared_types::PEDESTRIAN) {
                    const auto *ped_mean = category_size_stats::ped_mean_whl;
                    bounding_box_3d.tail<3>() = Eigen::Vector3d(ped_mean[0], ped_mean[1], ped_mean[2]);
                }
                else if (category == SUN::shared_types::CAR) {
                    const auto *car_mean = category_size_stats::car_mean_whl;
                    bounding_box_3d.tail<3>() = Eigen::Vector3d(car_mean[0], car_mean[1], car_mean[2]);
                }
                else if (category == SUN::shared_types::CYCLIST) {
                    const auto *cyc_mean = category_size_stats::cyclist_mean_whl;
                    bounding_box_3d.tail<3>() = Eigen::Vector3d(cyc_mean[0], cyc_mean[1], cyc_mean[2]);
                }

                bounding_box_3d[1] -= bounding_box_3d[1]/2.0;
                return bounding_box_3d;
            }

            GOT::tracking::Observation AddProposalInfo(const GOT::tracking::Observation &observation, const GOT::segmentation::ObjectProposal &proposal, const SUN::utils::Camera &camera, double dt) {

                auto fused_observation = observation;

                // (Re-)set indicators
                fused_observation.set_proposal_3d_avalible(true);

                // (Re-)set bounding box 3d, based on 3d measurements.
                fused_observation.set_bounding_box_3d(proposal.bounding_box_3d());

                // (Re-)set gp-pose and covariance based on stereo cues.
                Eigen::Vector4d pose = proposal.pos3d();
                pose.head<3>() = camera.ground_model()->ProjectPointToGround(pose.head<3>()); // Project to the ground.
                fused_observation.set_footpoint(pose);
                fused_observation.set_covariance3d(proposal.pose_covariance_matrix());

                // Set point-cloud indices
                fused_observation.set_pointcloud_indices(proposal.pointcloud_indices(), camera.width(), camera.height());

                return fused_observation;
            }

            GOT::tracking::Observation DetectionToObservation(const std::vector<SUN::utils::Detection> &detections, int det_id, const cv::Mat &image) {
                const auto &det = detections.at(det_id);

                // Init observation
                GOT::tracking::Observation obs;

                const Eigen::Vector4d &detection_bbox_2d = det.bounding_box_2d();
                const double detection_score = det.score();
                Eigen::Vector4d detection_footpoint = det.footpoint();
                Eigen::Matrix3d detection_cov3d = det.pose_covariance_matrix();
                Eigen::Matrix2d detection_cov2d;
                detection_cov2d(0, 0) = detection_cov3d(0, 0); detection_cov2d(0, 1) = detection_cov3d(0, 2);
                detection_cov2d(1, 0) = detection_cov3d(2, 0); detection_cov2d(1, 1) = detection_cov3d(2, 2);

                // Binary indicators
                obs.set_detection_avalible(true);
                obs.set_proposal_3d_avalible(false);

                // Bounding-box
                obs.set_bounding_box_2d(detection_bbox_2d);

                // Det. category, score, id
                auto det_category = static_cast<SUN::shared_types::CategoryTypeKITTI>(det.category());
                obs.set_detection_category(det_category);
                obs.set_detection_score(detection_score);

                // Color histogram
                obs.set_color_histogram(SUN::utils::observations::ComputeColorHistogramNOMT(image, detection_bbox_2d, 4));

                // Footpoint + uncertainty cov. mat.
                obs.set_footpoint(detection_footpoint);
                obs.set_covariance3d(detection_cov3d);

                // Bounding-box 3d and convex hull (here: use category-based prior)
                Eigen::VectorXd bounding_box_3d = LabelToBoundingBox3D(det_category, detection_footpoint);
                obs.set_bounding_box_3d(bounding_box_3d);
                obs.set_score(det.score());

                return obs;
            }

            std::vector<GOT::tracking::Observation> DetectionToSegmentationFusion(
                    const std::vector<SUN::utils::Detection> &detections_in,
                    const std::vector<GOT::segmentation::ObjectProposal> &proposals_in,
                    const SUN::utils::Camera &camera,
                    const cv::Mat &image,
                    const po::variables_map &parameter_map) {

                auto VectorIntersection = [](const std::vector<int> &indices_1, const std::vector<int> &indices_2)->int {
                    auto inds_p1 = indices_1;
                    auto inds_p2 = indices_2;
                    std::sort(inds_p1.begin(), inds_p1.end());
                    std::sort(inds_p2.begin(), inds_p2.end());
                    std::vector<int> set_intersection(std::max(inds_p1.size(), inds_p2.size()));
                    auto it_intersect=std::set_intersection(inds_p1.begin(), inds_p1.end(), inds_p2.begin(), inds_p2.end(), set_intersection.begin());
                    set_intersection.resize(it_intersect-set_intersection.begin());
                    return static_cast<int>(set_intersection.size());
                };

                std::vector<Observation> observations;
                std::vector<Observation> observations_det_supported;
                std::vector<Observation> observations_proposals_only;

                /// Holds association info for each enumeration.
                std::vector<AssocContext> observation_fusion_association_context;
                observation_fusion_association_context.reserve(proposals_in.size());

                // Algorithm:
                // 1. Enumerate all 'possible' associations
                // 2. Create interaction matrix
                // 3. Solve interaction matrix
                // 4. Take surviving hypos
                // 5. Post-processing:
                //      - Allow non-associated detections and proposals to make it to the next stage, too.

                /// Params
                const double e1 = parameter_map.at("observations_e1").as<double>(); // Minimal score
                const double e2 = parameter_map.at("observations_e2").as<double>(); // Overlap
                const double e3 = parameter_map.at("observations_e3").as<double>(); // Claiming same resources
                const double min_assoc_score = parameter_map.at("observations_min_association_score").as<double>();


                // -------------------------------------------------------------------------------
                // +++ ENUMERATE ASSOCIATIONS +++
                // -------------------------------------------------------------------------------

                /// Goes through all (detection, proposal) pairs, make candidates and compute association scores.

                // Sort detections by score
                auto detections_sorted_by_score = detections_in;
                std::sort(detections_sorted_by_score.begin(), detections_sorted_by_score.end(),
                          [](const SUN::utils::Detection &d1, const SUN::utils::Detection &d2) { return d1.score() > d2.score(); });


                for (int i = 0; i < detections_sorted_by_score.size(); i++) {
                    /// Init observation from the detection.
                    GOT::tracking::Observation detection_spawned_observation;
                    const auto &det = detections_sorted_by_score[i];
                    detection_spawned_observation = DetectionToObservation(detections_sorted_by_score, i, image);

                    /// 'Project' pose cov. matrix to the ground.
                    Eigen::Matrix3d detection_cov3d = det.pose_covariance_matrix();
                    Eigen::Matrix2d detection_cov2d;
                    detection_cov2d(0, 0) = detection_cov3d(0, 0); detection_cov2d(0, 1) = detection_cov3d(0, 2);
                    detection_cov2d(1, 0) = detection_cov3d(2, 0); detection_cov2d(1, 1) = detection_cov3d(2, 2);

                    /// Compute detection<->proposal association scores.
                    std::vector<AssocContext> assoc_scores(proposals_in.size()); // Holds detection-to-proposal assoc. info
                    for (int j = 0; j<proposals_in.size(); j++) {
                        const auto &prop = proposals_in.at(j);
                        double assoc_score = ComputeAssociationScore(prop, det.footpoint().head<3>(),
                                                                     detection_cov2d,
                                                                     det.bounding_box_2d(),
                                                                     static_cast<SUN::shared_types::CategoryTypeKITTI>(det.category()),
                                                                     camera, parameter_map);

                        assoc_scores.at(j).score = assoc_score;
                        assoc_scores.at(j).det_idx = i;
                        assoc_scores.at(j).proposal_idx = j;
                    }

                    /// Sort association context by score.
                    std::sort(assoc_scores.begin(), assoc_scores.end(), [](const AssocContext &p1, const AssocContext &p2) { return p1.score > p2.score; });

                    /// Access best association.
                    double best_assoc_score = 0.0;
                    if (assoc_scores.size() > 0) {
                        best_assoc_score = assoc_scores.front().score;
                    }

                    /// Merge detection and proposal-3D info
                    if (best_assoc_score>min_assoc_score) {

                        for (int prop_index=0; prop_index<assoc_scores.size(); prop_index++) {
                            auto assoc_context = assoc_scores.at(prop_index);
                            auto assoc_index =  assoc_context.proposal_idx;
                            auto assoc_score =  assoc_context.score;
                            if (assoc_score > min_assoc_score) {

                                const auto &best_assoc_proposal = proposals_in.at(assoc_index);

                                double dt = parameter_map.at("dt").as<double>();
                                auto fused_observation = AddProposalInfo(detection_spawned_observation, best_assoc_proposal, camera, dt);

                                // Push to the list!
                                observations_det_supported.push_back(fused_observation);
                                observation_fusion_association_context.push_back(assoc_context);
                            }
                        }
                    }
                }

                // -------------------------------------------------------------------------------
                // +++ ENCODE GRAPH TO THE MATRIX +++
                // -------------------------------------------------------------------------------

                Eigen::MatrixXd Q;
                const auto M = observations_det_supported.size();
                Q.setZero(M, M);

                /// Unaries => association scores.
                for (int i=0; i<M; i++)
                    Q(i,i) = -e1 + observation_fusion_association_context.at(i).score;

                /// Pairwise => compute interactions between enumerations.
                // - (e2*hard_exclusion_for_shared_data + e3*physical_overlap)

                for (int i=0; i<M; i++) {
                    for (int j=i+1; j<M; j++) {
                        const auto &obs_i = observations_det_supported.at(i);
                        const auto &obs_j = observations_det_supported.at(j);
                        const auto &assoc_context_i = observation_fusion_association_context.at(i);
                        const auto &assoc_context_j = observation_fusion_association_context.at(j);

                        // Early reject-check: are bboxes overlapping at all?
                        // Note: the higher the IOU threshold, the sparser the problem!
                        if (SUN::utils::bbox::IntersectionOverUnion2d(obs_i.bounding_box_2d(), obs_j.bounding_box_2d()) < 0.05)
                            continue;

                        // Lets check the following:
                        // * Are resources sharing the 'data' (detections or proposals)
                        // * Physical overlap: pixel indices

                        bool observations_share_data = false;

                        // Shared detection?
                        if (assoc_context_i.det_idx==assoc_context_j.det_idx)
                            observations_share_data = true;

                        // Shared proposal?
                        if (assoc_context_i.proposal_idx==assoc_context_j.proposal_idx)
                            observations_share_data = true;

                        // Compute mask overlap penalty (based on mask indices)
                        int intersection_size = VectorIntersection(obs_i.pointcloud_indices(), obs_j.pointcloud_indices());
                        int min_indices_set_size = std::min(obs_i.pointcloud_indices().size(), obs_j.pointcloud_indices().size());
                        double physical_overlap = 0.0;
                        if (obs_i.pointcloud_indices().size() > 0 && obs_j.pointcloud_indices().size()>0)
                            physical_overlap = static_cast<double>(intersection_size) / static_cast<double>(min_indices_set_size);

                        double overlap_penalty = - (e2*physical_overlap + (observations_share_data?e3:0.0));

                        Q(i,j) = overlap_penalty;
                        Q(j,i) = overlap_penalty;
                    }
                }

                // -------------------------------------------------------------------------------
                // +++ INFERENCE +++
                // -------------------------------------------------------------------------------
                Eigen::VectorXi m;
                QPBO::SolveMultiBranch(Q, m);

                // -------------------------------------------------------------------------------
                // +++ POSTPROCESSING +++
                // -------------------------------------------------------------------------------
                std::set<int> det_indices; // Save here indices of 'fused' detections
                std::vector<Observation> accepted_observations;

                /// Find out which detections were not selected and associated to proposals
                for (int i=0; i<m.size(); i++) {
                    if (m[i] == 1) {
                        accepted_observations.push_back(observations_det_supported.at(i));
                        int det_id = observation_fusion_association_context.at(i).det_idx;
                        det_indices.insert(det_id);
                    }
                }

                /// At those to the observation set (=> want to use detections, even if they can not be precisely localized in 3D),
                for (int i=0; i<detections_in.size(); i++) {
                    if (det_indices.count(i)<=0) {
                        // Detection was not fused with proposal => insert 'raw' detection to the observation set.
                        auto detection_spawned_observation = DetectionToObservation(detections_in, i, image);
                        accepted_observations.push_back(detection_spawned_observation);
                    }
                }

                return accepted_observations;
            }

            std::vector<GOT::tracking::Observation> DetectionsOnly (
                    const std::vector <SUN::utils::Detection> &detections_in,
                    const std::vector<GOT::segmentation::ObjectProposal> &proposals_in,
                    const cv::Mat &image) {

                std::vector<GOT::tracking::Observation> observations;

                // Easy case: just copy detection info to ObjectProposal.
                for (int i = 0; i < detections_in.size(); i++) {
                    auto detection_spawned_observation = DetectionToObservation(detections_in, i, image);
                    observations.push_back(detection_spawned_observation);
                }

                return observations;
            }
        }
    }
}

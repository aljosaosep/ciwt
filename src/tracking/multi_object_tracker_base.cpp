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

#include <tracking/multi_object_tracker_base.h>

// utils
#include "utils_bounding_box.h"
#include "ground_model.h"
#include "camera.h"

namespace GOT {
    namespace tracking {

        MultiObjectTracker3DBase::MultiObjectTracker3DBase(const po::variables_map &params) {
            this->parameter_map_ = params;
            last_hypo_id_ = 0;
        }

        void MultiObjectTracker3DBase::CheckExitZones(const SUN::utils::Camera &camera, int current_frame) {

            assert(parameter_map_.count("tracking_exit_zones_lateral_distance"));
            assert(parameter_map_.count("tracking_exit_zones_rear_distance"));
            assert(parameter_map_.count("tracking_exit_zones_far_distance"));

            double dist_lat = parameter_map_.at("tracking_exit_zones_lateral_distance").as<double>();
            double dist_rear = parameter_map_.at("tracking_exit_zones_rear_distance").as<double>();
            double dist_far = parameter_map_.at("tracking_exit_zones_far_distance").as<double>();

            const double multi=0.8;
            SUN::utils::Ray ray_top_left, ray_top_right, ray_bottom_left, ray_bottom_right;

            // Cast rays from camera origin through bottom-left, bottom-right, nearly-bottom-left, nearly-bottom-right pixels.
            ray_top_left = camera.GetRayCameraSpace(1, camera.height());
            ray_top_right = camera.GetRayCameraSpace(camera.width()-1, camera.height());
            ray_bottom_left = camera.GetRayCameraSpace(1, static_cast<int>(camera.height()*multi));
            ray_bottom_right = camera.GetRayCameraSpace(camera.width()-1, static_cast<int>(camera.height()*multi));

            // Intersect those rays with a ground plane. This will give us points, that define left and right 'border' of the viewing frustum
            Eigen::Vector3d proj_top_left = camera.ground_model()->IntersectRayToGround(ray_top_left.origin, ray_top_left.direction);
            Eigen::Vector3d proj_top_right = camera.ground_model()->IntersectRayToGround(ray_top_right.origin, ray_top_right.direction);
            Eigen::Vector3d proj_bottom_left = camera.ground_model()->IntersectRayToGround(ray_bottom_left.origin, ray_bottom_left.direction);
            Eigen::Vector3d proj_bottom_right = camera.ground_model()->IntersectRayToGround(ray_bottom_right.origin, ray_bottom_right.direction);

            // Compute vectors, defined by left and right points.
            Eigen::Vector3d left_vec = proj_bottom_left - proj_top_left;
            Eigen::Vector3d right_vec = proj_bottom_right - proj_top_right;

            /*
                  Camera frustum: left plane, right plane, bottom plane
                  Planes are defined by plane normals and 'd' ([a b c d] parametrization)

                                       \->    <-/
                                        \      /
                                         \  | /
                                          ----
             */

            // Camera frustum 'left' and 'right' planes
            Eigen::Vector3d normal_left = camera.ground_model()->Normal(proj_bottom_left);
            Eigen::Vector3d normal_right = camera.ground_model()->Normal(proj_bottom_right);

            Eigen::Vector3d normal_left_plane = left_vec.cross(normal_left).normalized();
            Eigen::Vector3d normal_right_plane = right_vec.cross(normal_right).normalized() * -1.0;
            double d_left_plane = -1.0*normal_left_plane.dot(proj_top_left);
            double d_right_plane = -1.0*normal_right_plane.dot(proj_top_right);

            Eigen::Vector3d normal_principal_plane = Eigen::Vector3d(0.0, 0.0, 1.0);
            double d_principal_plane = 0.0;


            int end_tolerance = 0; //1; // How many points need to be behind the exit zone(s)?

            // For each hypothesis, check if it is out of camera frustrum bounds!
            for (auto &hypo:hypotheses_) {
                if (hypo.poses().size() < 3)
                    continue;

                if (hypo.terminated())
                    continue;

                // Don't terminate newly created hypos
                if (current_frame-hypo.creation_timestamp_ < 10)
                    continue;

                // Count how many trajectory points falls out of bounds to these variables.
                int left_plane_count = 0;
                int right_plane_count = 0;
                int bottom_plane_count = 0;
                //int far_plane_count = 0;
                int far_count = 0;

                // Check last three traj. points
                for(int j=hypo.poses().size()-1; j>(hypo.poses().size()-3); j--) {
                    auto hypo_point = hypo.poses().at(j);
                    hypo_point = camera.WorldToCamera(hypo_point);

                    // Dennis magic ...
                    double left_plane_check = hypo_point[0]*normal_left_plane[0] + hypo_point[1]*normal_left_plane[1] + hypo_point[2]*normal_left_plane[2] + d_left_plane;
                    double right_plane_check = hypo_point[0]*normal_right_plane[0] + hypo_point[1]*normal_right_plane[1] + hypo_point[2]*normal_right_plane[2] + d_right_plane;
                    double bottom_plane_check = hypo_point[0]*normal_principal_plane[0] + hypo_point[1]*normal_principal_plane[1] + hypo_point[2]*normal_principal_plane[2] + d_principal_plane;

                    if (left_plane_check < dist_lat)
                        left_plane_count ++;
                    if (right_plane_check < dist_lat)
                        right_plane_count ++;
                    if (bottom_plane_check < dist_rear)
                        bottom_plane_count ++;

                    if (hypo_point[2] > dist_far)
                        far_count ++;
                }

                if(left_plane_count > end_tolerance) {
                    if (verbose_) std::cout << "Hypo "<< hypo.id() << " entered LEFT EXIT ZONE." << std::endl;
                    hypo.set_terminated(true);
                }
                else if(right_plane_count > end_tolerance) {
                    if (verbose_) std::cout << "Hypo "<< hypo.id() << " entered RIGHT EXIT ZONE." << std::endl;
                    hypo.set_terminated(true);
                }
                else if (bottom_plane_count > end_tolerance /*|| (far_plane_count > 2)*/) {
                    if (verbose_) std::cout << "Hypo "<< hypo.id() << " entered BACK EXIT ZONE." << std::endl;
                    hypo.set_terminated(true);
                }

                if (far_count > end_tolerance) {
                    if (verbose_) std::cout << "Hypo "<< hypo.id() << " went past FAR EXIT ZONE." << std::endl;
                    hypo.set_terminated(true);
                }
            }
        }

        void MultiObjectTracker3DBase::AssignIDs (const Eigen::VectorXi &m, Resources::ConstPtr detections,
                                                  int current_frame, std::vector<Hypothesis> &hypotheses) {

            assert(parameter_map_.count("id_handling_overlap_threshold"));
            double overlap_thresh = parameter_map_.at("id_handling_overlap_threshold").as<double>();

            if (verbose_) printf("\nID handling ...\n");
            // Let's cache indices of currently selected hypos. These should not be re-assigned.
            std::set<int> dont_reassign_these_hypo_ids;
            for (int i = 0; i < m.size(); i++) {
                int is_hypo_selected = m[i];
                int id = hypotheses.at(i).id();
                if (static_cast<bool>(is_hypo_selected) && (id > -1))
                    dont_reassign_these_hypo_ids.insert(id);
            }

            for (int i = 0; i < m.size(); i++) {
                int is_hypo_selected = m[i];
                if (is_hypo_selected) { // Only assign id's to selected.
                    Hypothesis &hypo = hypotheses.at(i);
                    if (hypo.id() >= 0) {
                        // Trivial case... only update the stack.
                        for (auto &hypo_in_stack:this->hypo_stack_) {
                            if (hypo_in_stack.id() == hypo.id()) {
                                hypo_in_stack = hypo;
                            }
                        }
                    }
                    else {// Uh-oh, hypo has no id... see if we find a match in active hypos stack!
                        int num_intersect_best = 0;
                        int hypo_id_new = -1;
                        int index_best = -1;
                        //for (auto &hypo_in_stack:this->hypo_stack_) {
                        for (int stack_index = 0; stack_index < hypo_stack_.size(); stack_index++) {
                            auto &hypo_in_stack = this->hypo_stack_.at(stack_index);
                            // Check if we can re-assign this hypo in stack.

                            // Ignore, if this hypo is already in the selected set
                            if ((dont_reassign_these_hypo_ids.size() > 0) &&
                                (dont_reassign_these_hypo_ids.count(hypo_in_stack.id()) > 0))
                                continue;

                            // Compute intersection and overlap fraction!
                            std::vector<HypothesisInlier> intersect = IntersectInliers(hypo,
                                                                                       hypo_in_stack,
                                                                                       detections,
                                                                                       current_frame); //IntersectInliers(hypo, hypo_in_stack);
                            int num_obs_intersect = intersect.size();
                            int num_obs_old = hypo_in_stack.inliers().size();
                            int num_obs_new = hypo.inliers().size();
                            double frac = 0.0;
                            if (num_obs_intersect > 0) {
                                frac = static_cast<double>(num_obs_intersect) /
                                       std::min(static_cast<double>(num_obs_old),
                                                static_cast<double>(num_obs_new));
                            }
                            if ((frac > overlap_thresh) && // frac > 0.5
                                // We take the hypo with longest intersection (not the one with best intersect. fraction!)
                                (num_obs_intersect > num_intersect_best)) {
                                hypo_id_new = hypo_in_stack.id();
                                num_intersect_best = num_obs_intersect;
                                index_best = stack_index;
                            }
                        }
                        // If no id was "inherited", then gen. a new one.
                        if (hypo_id_new < 0) {
                            //std::cout << "\33[36;40;1m" << " New hypo with id: " << this->last_hypo_id_  << "\33[0m" << std::endl;
                            if (verbose_)  printf("New hypo with id: %d\n", this->last_hypo_id_);

                            hypo.set_id(this->last_hypo_id_++);

                            hypo_stack_.push_back(hypo);  // Add new hypo to the stack
                            dont_reassign_these_hypo_ids.insert((hypo_id_new - 1));
                        }
                        else {
                            //std::cout << "\33[36;40;1m" << " Replacing hypo with id: " << hypo_id_new <<  "\33[0m" << std::endl;
                            if (verbose_) printf("Replacing hypo with id: %d\n", hypo_id_new);

                            // We found overlapping active hypo. Let's assign id and update the stack!
                            hypo.set_id(hypo_id_new);
                            hypo_stack_.at(index_best) = hypo;

                            // We assigned this id to selected hypo -- we can't assign it again!
                            dont_reassign_these_hypo_ids.insert(hypo_id_new);

                            // Make sure old hypo with the same id is gone.
                            // This impl. just sets index to -1, so hypo has chance to be selected again. Alternatively, it could be removed.
                            for (int k = 0; k < hypotheses.size(); k++) {
                                if (!static_cast<bool>(m[k]) && hypotheses.at(k).id() ==
                                                                hypo_id_new) { // Need to be NOT selected, and have same id
                                    hypotheses.at(k).set_id(-1);
                                }
                            }
                        }
                    }
                }
            }
        }

        void MultiObjectTracker3DBase::SetLastFrameSelected(std::vector<Hypothesis> &hypotheses, const Eigen::VectorXi &m,
                                                            int frame) {
            assert(hypotheses.size() == m.size());

            for (int i = 0; i < m.size(); i++) {
                int is_hypo_selected = m[i];
                if (is_hypo_selected) {
                    hypotheses.at(i).set_last_frame_selected(frame);
                }
            }
        }

        std::vector<Hypothesis> MultiObjectTracker3DBase::RemoveInactiveHypotheses(const std::vector<Hypothesis> &hypotheses,
                                                                                   int frame, int num_frames_not_selected_threshold) {
            std::vector<Hypothesis> filtered_set;
            for (const auto &hypo:hypotheses) {
                if (std::abs(hypo.last_frame_selected() - frame) <=
                    num_frames_not_selected_threshold)
                    filtered_set.push_back(hypo);
            }
            return filtered_set;
        }


        std::vector<HypothesisInlier> MultiObjectTracker3DBase::IntersectInliers(const Hypothesis &hypothesis_1,
                                                                                 const Hypothesis &hypothesis_2,
                                                                                 Resources::ConstPtr detections,
                                                                                 int frame) {
            // Works as follows:
            // Loop through inliers
            // if indices match
            //      put them to intersection list
            // return intersection list

            const std::vector<HypothesisInlier> &hypo_inliers_1 = hypothesis_1.inliers();
            const std::vector<HypothesisInlier> &hypo_inliers_2 = hypothesis_2.inliers();

            // ================= FOR EFFICIENT INTERSECT. =====================================
            int inliers_1_first_stamp = hypo_inliers_1.front().timestamp_;
            int inliers_1_back_stamp = hypo_inliers_1.back().timestamp_;
            int inliers_2_first_stamp = hypo_inliers_2.front().timestamp_;
            int inliers_2_back_stamp = hypo_inliers_2.back().timestamp_;
            int first_timestamp = std::max(inliers_1_first_stamp, inliers_2_first_stamp);
            int last_timestamp = std::min(inliers_1_back_stamp, inliers_2_back_stamp);
            // ==========================================================================

            std::vector<HypothesisInlier> intersection;
            if (hypo_inliers_1.size() > 0 && hypo_inliers_2.size() > 0) {

                //! Warning: Quadratic in #inliers!
                for (const auto &inlier_1:hypo_inliers_1) {
                    const int t_inlier_1 = inlier_1.timestamp_;
                    // -----------------------------------------
                    if (t_inlier_1 < first_timestamp)
                        continue;
                    if (t_inlier_1 > last_timestamp)
                        continue;
                    // -----------------------------------------
                    for (const auto &inlier_2:hypo_inliers_2) {
                        const int t_inlier_2 = inlier_2.timestamp_;

                        // -----------------------------------------
                        if (t_inlier_2 < first_timestamp)
                            continue;
                        if (t_inlier_2 > last_timestamp)
                            continue;
                        // -----------------------------------------

                        if ((t_inlier_1 == t_inlier_2) &&
                            (inlier_1.index_ == inlier_2.index_)) {
                            intersection.push_back(inlier_1); // inlier_2 is the same.
                        }
                    }
                }

            }

            return intersection;
        }

        // Setters / getters
        const HypothesesVector &MultiObjectTracker3DBase::selected_hypotheses() const {
            return selected_set_;
        }

        const std::vector<Hypothesis> &MultiObjectTracker3DBase::terminated_hypotheses() const {
            return terminated_hypotheses_;
        }

        void MultiObjectTracker3DBase::set_verbose(bool verbose) {
            verbose_ = verbose;
        }
    }
}

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

#include "potential_functions.h"

// utils
#include "sun_utils/utils_bounding_box.h"

namespace GOT {
    namespace tracking {
        namespace CRF {

            typedef std::function<double(const GOT::tracking::Hypothesis &, const GOT::tracking::Hypothesis &, int)> OverlapFnc;

            namespace CIWT {
                namespace hypothesis_selection {
                    
                    namespace overlap {
                        double OverlapFncRectangleIoU(const GOT::tracking::Hypothesis &hypothesis_1,
                                                      const GOT::tracking::Hypothesis &hypothesis_2, int frame) {
                            const auto &bb1 = hypothesis_1.bounding_boxes_2d_with_timestamp().at(frame);
                            const auto &bb2 = hypothesis_2.bounding_boxes_2d_with_timestamp().at(frame);
                            double IOU = SUN::utils::bbox::IntersectionOverUnion2d(bb1, bb2);
                            return IOU * IOU; // Return squared IoU
                        }
                    }

                    namespace potential_func {
                        double HypoUnary (const Hypothesis &hypo, int current_frame, const po::variables_map &parameter_map) {

                            double e1 = parameter_map.at("tracking_e1").as<double>();
                            double hole_penalty_decay_parameter = parameter_map.at("hole_penalty_decay_parameter").as<double>();
                            int tau = parameter_map.at("tracking_exp_decay").as<int>();

                            std::vector<HypothesisInlier> hypo_inliers = hypo.inliers();
                            double hypo_score = 0.0;

                            for (const HypothesisInlier &inlier:hypo_inliers) {
                                double association_score = inlier.association_score_;
                                double inlier_score = inlier.inlier_score_;
                                int frame_of_detection = inlier.timestamp_;
                                // compute temporal decay: decay = std::exp( -(current_frame-inlier_frame) / tau )
                                const double exp_decay = std::exp(-1.0 * (static_cast<double>(current_frame - frame_of_detection)) / tau);
                                double score = exp_decay * association_score * inlier_score;
                                hypo_score += score;
                            }

                            // Exp. penalty for holes
                            const int last_inlier_timestamp = hypo.inliers().back().timestamp_;
                            double hole_penalty = std::exp(-(static_cast<double>(current_frame - last_inlier_timestamp) / hole_penalty_decay_parameter));
                            return -e1 + (hole_penalty * hypo_score);
                        }

                        double HypoPairwise(const Hypothesis &hypo_1, const Hypothesis &hypo_2, int current_frame, const po::variables_map &parameter_map) {
                            double physical_overlap_penalty = ComputePairwiseOverlap(
                                    hypo_1, hypo_2, current_frame, overlap::OverlapFncRectangleIoU
                            );

                            auto det_intersection = IntersectInliersDefault(hypo_1, hypo_2, current_frame);
                            double overlap_penalty = 0.0;
                            double param_e3 = parameter_map.at("tracking_e3").as<double>(); // 'soft' penalty
                            double param_e4 = parameter_map.at("tracking_e4").as<double>(); // 'hard' penalty (inlier intersect.)
                            if (physical_overlap_penalty > 0.001 || det_intersection.size() > 0)
                                overlap_penalty = param_e3 * physical_overlap_penalty + param_e4 * det_intersection.size();

                            return overlap_penalty;
                        }
                    }

                    std::vector<GOT::tracking::HypothesisInlier> IntersectInliersDefault(
                            const GOT::tracking::Hypothesis &hypothesis_1,
                            const GOT::tracking::Hypothesis &hypothesis_2,
                            int frame) {
                        // Works as follows:
                        // Loop through inliers
                        // if indices match
                        //      put them to intersection list
                        // return intersection list

                        const std::vector<GOT::tracking::HypothesisInlier> &hypo_inliers_1 = hypothesis_1.inliers();
                        const std::vector<GOT::tracking::HypothesisInlier> &hypo_inliers_2 = hypothesis_2.inliers();

                        int inliers_1_first_stamp = hypo_inliers_1.front().timestamp_;
                        int inliers_1_back_stamp = hypo_inliers_1.back().timestamp_;
                        int inliers_2_first_stamp = hypo_inliers_2.front().timestamp_;
                        int inliers_2_back_stamp = hypo_inliers_2.back().timestamp_;
                        int first_timestamp = std::max(inliers_1_first_stamp, inliers_2_first_stamp);
                        int last_timestamp = std::min(inliers_1_back_stamp, inliers_2_back_stamp);

                        std::vector<GOT::tracking::HypothesisInlier> intersection;
                        if (hypo_inliers_1.size() > 0 && hypo_inliers_2.size() > 0) {
                            for (const auto &inlier_1:hypo_inliers_1) {
                                const int t_inlier_1 = inlier_1.timestamp_;

                                // Early reject
                                if (t_inlier_1 < first_timestamp)
                                    continue;
                                if (t_inlier_1 > last_timestamp)
                                    continue;

                                for (const auto &inlier_2:hypo_inliers_2) {
                                    const int t_inlier_2 = inlier_2.timestamp_;

                                    // Early reject
                                    if (t_inlier_2 < first_timestamp)
                                        continue;
                                    if (t_inlier_2 > last_timestamp)
                                        continue;

                                    // Matching index + matching timestamp => add to the intersection list
                                    if ((t_inlier_1 == t_inlier_2) && (inlier_1.index_ == inlier_2.index_)) {
                                        intersection.push_back(inlier_1); // inlier_2 is the same.
                                    }
                                }
                            }
                        }

                        return intersection;
                    }


                    double ComputePairwiseOverlap(const GOT::tracking::Hypothesis &hypothesis_1, const GOT::tracking::Hypothesis &hypothesis_2,
                                                  int frame, OverlapFnc overlap_fnc) {

                        const auto &timestamps_1 = hypothesis_1.timestamps();
                        const auto &timestamps_2 = hypothesis_2.timestamps();

                        int hypo_1_first_stamp = timestamps_1.front();
                        int hypo_1_back_stamp = timestamps_1.back();
                        int hypo_2_first_stamp = timestamps_2.front();
                        int hypo_2_back_stamp = timestamps_2.back();
                        int first_timestamp = std::max(hypo_1_first_stamp, hypo_2_first_stamp);
                        int last_timestamp = std::min(hypo_1_back_stamp, hypo_2_back_stamp);

                        double intersection_sum = 0.0;
                        if (timestamps_1.size() > 0 && timestamps_2.size() > 0) {

                            for (const auto &t_inlier_1:timestamps_1) {
                                // Early reject
                                if (t_inlier_1 < first_timestamp)
                                    continue;
                                if (t_inlier_1 > last_timestamp)
                                    continue;

                                for (const auto &t_inlier_2:timestamps_2) {

                                    // Early reject
                                    if (t_inlier_2 < first_timestamp)
                                        continue;
                                    if (t_inlier_2 > last_timestamp)
                                        continue;

                                    if (t_inlier_1 == t_inlier_2) {
                                        // -------------------- Compute overlap -----------------------------------
                                        const double exp_decay = 1.0; // At the moment, dummy
                                        intersection_sum += overlap_fnc(hypothesis_1, hypothesis_2, frame)*exp_decay;
                                        // ------------------------------------------------------------------------
                                    }
                                }
                            }
                            return intersection_sum;
                        }
                        return 0.0; // No intersection whatsoever
                    }
                }
            }
        }
    }
}

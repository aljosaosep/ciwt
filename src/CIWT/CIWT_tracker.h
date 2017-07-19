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

#ifndef GOT_MULTICUE_OBJECT_TRACKER_H
#define GOT_MULTICUE_OBJECT_TRACKER_H

// std
#include <vector>
#include <memory>
#include <set>

// Tracking
#include <tracking/multi_object_tracker_base.h>

// Project
#include "CIWT_dynamics_handler.h"

namespace GOT {
    namespace tracking {
        namespace ciwt_tracker {

            /**
               * @brief This class implements ''Combined Image- and World-Space Tracking in Traffic Scenes'', ICRA'17
               * @author Aljosa (osep@vision.rwth-aachen.de)
               */
            class CIWTTracker : public MultiObjectTracker3DBase {
            public:

                CIWTTracker(const po::variables_map &params);

                /**
                   * @brief Tracker 'main': hypothesis set creation, selection, id-management etc.
                   */
                void ProcessFrame(Resources::ConstPtr detections, int current_frame);

                /**
                   * @brief Applies state prediction, data association and correction/extrapolation
                   */
                void AdvanceHypo(Resources::ConstPtr detections, int reference_frame, bool is_forward, Hypothesis &ref_hypo, bool allow_association = true);

                /**
                   * @brief Extends exiting hypothesis set with new observations
                   */
                std::vector<Hypothesis> ExtendHypotheses(const std::vector<Hypothesis> &current_hypotheses_set, Resources::ConstPtr detections, int current_frame);

                /**
                   * @brief Starts new hypotheses from the observations
                   */
                std::vector<Hypothesis> StartNewHypotheses(Resources::ConstPtr detections, int current_frame);

                /**
                   * @brief Initializes hypothesis state
                   */
                void HypothesisInit(Resources::ConstPtr detections, bool is_forward_update, int inlier_index, int current_frame, Hypothesis &hypo);

                /**
                   * @brief Updates hypothesis state
                   */
                void HypothesisUpdate(Resources::ConstPtr detections, bool is_forward_update, int inlier_index, int current_frame, Hypothesis &hypo);

                /**
                   * @brief Data association.
                   */
                bool AssociateObservationToHypothesis(const std::vector<Observation> &observations,
                                                      const SUN::utils::Camera &camera,
                                                      const Hypothesis &hypo,
                                                      std::vector<double> &observations_association_scores,
                                                      std::vector< std::vector<double> > &association_scores_debug);
            protected:
                std::unique_ptr<GOT::tracking::DynamicsModelHandler> dynamics_handler_;
                std::vector<Hypothesis> not_selected_hypos_;
            };

        }
    }
}

#endif

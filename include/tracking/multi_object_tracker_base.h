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

#ifndef GOT_MULTI_HYPOTHESIS_OBJECT_TRACKER_H
#define GOT_MULTI_HYPOTHESIS_OBJECT_TRACKER_H

// std
#include <vector>
#include <memory>
#include <set>

// Project
#include "resource_manager.h"
#include "hypothesis.h"

// boost
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace GOT {
    namespace tracking {

        /**
           * @brief Base class for trackers. Just an interface.
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class MultiObjectTracker3DBase {
        public:
            MultiObjectTracker3DBase(const po::variables_map &params);

            /**
               * @brief This should be the main tracing processing function.
               * @return Nothing. Updates internal member hypotheses_.
               */
            virtual void ProcessFrame(Resources::ConstPtr detections, int frame)=0;

            /**
               * @brief This function checks if trajectory of hypothesis is close to view frustum boundaries.
               *        If so, the termination flag is set.
               *
               * @return Nothing. Sets hypo state 'terminated' to true, if trajectory falls outside!
               */
            void CheckExitZones(const SUN::utils::Camera &camera, int current_frame);
            
            virtual void AssignIDs (const Eigen::VectorXi &m, Resources::ConstPtr detections, int current_frame,
                                    std::vector<Hypothesis> &hypotheses);
            
            void SetLastFrameSelected(std::vector<Hypothesis> &hypotheses, const Eigen::VectorXi &m, int frame);
            
            std::vector<Hypothesis> RemoveInactiveHypotheses(const std::vector<Hypothesis> &hypotheses,
                                                             int frame, int num_frames_not_selected_threshold);
            
            std::vector<HypothesisInlier> IntersectInliers(const Hypothesis &hypothesis_1,
                                                           const Hypothesis &hypothesis_2,
                                                           Resources::ConstPtr detections,
                                                           int frame);
            
            // Setters
            void set_verbose(bool verbose);

            // Getters
            const std::vector<Hypothesis> &selected_hypotheses() const;
            const std::vector<Hypothesis> &terminated_hypotheses() const;

        protected:

            std::vector<Hypothesis> hypotheses_;
            std::set<int> detection_indices_used_for_extensions_; // Store info, which detections were used for hypo extend.
            std::vector<Hypothesis> hypo_stack_; // Active hypos stack, for id handling.
            std::vector<Hypothesis> selected_set_; // Selected hypothesis set. Should be updated each frame.
            std::vector<Hypothesis> terminated_hypotheses_;

            int last_hypo_id_; // Last assigned id.
            po::variables_map parameter_map_;
            bool verbose_ = false;
        };
    }
}
#endif

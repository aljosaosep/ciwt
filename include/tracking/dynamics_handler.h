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

#ifndef GOT_DYNAMICS_HANDLER_H
#define GOT_DYNAMICS_HANDLER_H


// Eigen
#include <Eigen/Core>

// Tracking
#include <tracking/hypothesis.h>

// boost
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace GOT {
    namespace tracking {
            /**
               * @brief Base dynamics handler.
               *        Takes care of different filters, transition/observation models, their initialization etc.
               * @author Aljosa (osep@vision.rwth-aachen.de)
               */
            class DynamicsModelHandler {
            public:
                DynamicsModelHandler(const po::variables_map &params) {
                    parameters_ = params;
                }

                // Interface
                virtual void InitializeState(const SUN::utils::Camera &camera, const Observation &obs, bool forward, Hypothesis &hypo) = 0;
                virtual void ApplyTransition(const SUN::utils::Camera &camera, const Eigen::VectorXd &u, Hypothesis &hypo) = 0;
                virtual void ApplyCorrection(const SUN::utils::Camera &camera, const Observation &obs, bool forward, Hypothesis &hypo) = 0;

            protected:
                po::variables_map parameters_;
            };
    }
}

#endif //GOT_DYNAMICS_HANDLER_H

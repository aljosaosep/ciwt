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

#ifndef KALMAN_FILTER_HANDLER_CONST_VELOCITY_H
#define KALMAN_FILTER_HANDLER_CONST_VELOCITY_H

// eigen
#include <Eigen/Core>
#include <src/sun_utils/shared_types.h>

// tracking
#include <tracking/dynamics_handler.h>

// Forward declarations
namespace GOT { namespace tracking { class Hypothesis; }}
namespace SUN { namespace utils { class Camera; }}
namespace GOT { namespace tracking { class Observation; }}

namespace GOT {
    namespace tracking {
        /**
         * @brief This class implements 'original' CIWT dynamics handler, used in the ICRA'17 paper.
         *        Uses CVTR Kalman Filter (const-velocity-turn-rate).
         */
        class DynamicsHandlerCIWT : public DynamicsModelHandler {
        public:
            DynamicsHandlerCIWT(const po::variables_map &params);

            /// Interface methods impl.

            /**
             * @brief Initialize state, either use 3D segm. info or category priors, if 3D segm. is not avalible.
             */
            void InitializeState(const SUN::utils::Camera &camera, const Observation &obs, bool forward, Hypothesis &hypo);

            /**
             * @brief Applies KF prediction.
             */
            void ApplyTransition(const SUN::utils::Camera &camera, const Eigen::VectorXd &u, Hypothesis &hypo);

            /**
             * @brief Applies KF correction using one of the observation models.
             */
            void ApplyCorrection(const SUN::utils::Camera &camera, const Observation &obs, bool forward, Hypothesis &hypo);

            /**
             * @brief Initializes prior and system noise matrices, different handling for 'car' and 'pedestrian'.
             */
            void InitializeMatrices();


            /// Initial uncertainties
            Eigen::MatrixXd P0_pedestrian_;
            Eigen::MatrixXd P0_car_;

            /// System noise
            Eigen::MatrixXd G_pedestrian_;
            Eigen::MatrixXd G_car_;

            /// Observation matrices
            Eigen::MatrixXd H_observe_pos_bbox2d_size_;
            Eigen::MatrixXd H_observe_everything_;

        private:
            /// 2D bounding box noise
            double bounding_box_obs_noise_;
            double bounding_box_system_noise_;
            double bounding_box_initial_variance_;

            /// Initial uncertainties for velocity
            Eigen::Vector2d P0_car_velocity_variance_;
            Eigen::Vector2d P0_ped_velocity_variance_;

            /// System noise - pose and velocity
            Eigen::Vector2d G_car_velocity_variance_;
            Eigen::Vector2d G_ped_velocity_variance_;
            Eigen::Vector2d G_car_pos_variance_;
            Eigen::Vector2d G_ped_pos_variance_;

            int state_dim_;
            double max_dist_use_stereo_;
            bool enable_kalman_coupling_;
        };
    }
}

#endif

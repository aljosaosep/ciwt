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

#ifndef GOT_EXTENDED_KALMAN_FILTER_H
#define GOT_EXTENDED_KALMAN_FILTER_H

// std
#include <memory>

// Eigen
#include <Eigen/Core>

#include "kalman_filter.h"

namespace GOT {
    namespace tracking {

        /**
           * @brief Base class for EKF. Just an interface.
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class ExtendedKalmanFilter : public KalmanFilter {
        public:
            ExtendedKalmanFilter(const Parameters &params);

            /**
               * @brief This method should set the initial state.
               */
            virtual void Init(const Eigen::VectorXd &x_0, const Eigen::MatrixXd &P_0)=0;


            // In case we have EKF.
            // Implement your non-linear state trans. func. in case you like extended kalman filters.
            // Don't forget that your transition matrix should then be Jacobian of your non-linear state transition func.
            virtual Eigen::VectorXd NonlinearStateProjection() const = 0;
            virtual Eigen::MatrixXd LinearizeTransitionMatrix() const = 0;
            virtual Eigen::VectorXd compute_measurement_residual(const Eigen::VectorXd z_t, const Eigen::MatrixXd &H)=0;

            virtual void ComputePrediction(const Eigen::VectorXd &u, Eigen::VectorXd &x_prio, Eigen::MatrixXd &P_prio) const;

            void Correction(const Eigen::VectorXd z_t, const Eigen::MatrixXd &observation_cov_t, const Eigen::MatrixXd &H);

            // Typedefs
            typedef std::shared_ptr<const ExtendedKalmanFilter> ConstPtr;
            typedef std::shared_ptr<ExtendedKalmanFilter> Ptr;
        };

        /**
           * @brief Base class for the coupled-state filters.
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class CoupledExtendedKalmanFilter : public ExtendedKalmanFilter {
        public:
            CoupledExtendedKalmanFilter (const Parameters &params);


            virtual void Init(const Eigen::VectorXd &x_0, const Eigen::MatrixXd &P_0)=0;


            // In case we have EKF.
            // Implement your non-linear state trans. func. in case you like extended kalman filters.
            // Don't forget that your transition matrix should then be Jacobian of your non-linear state transition func.
            virtual Eigen::VectorXd NonlinearStateProjection() const = 0;
            virtual Eigen::MatrixXd LinearizeTransitionMatrix() const = 0;
            virtual Eigen::VectorXd compute_measurement_residual(const Eigen::VectorXd z_t, const Eigen::MatrixXd &H)=0;


            // These need to be re-set for every frame
            void set_camera_current_frame(const SUN::utils::Camera &camera);
            void set_use_2d_measurement_to_update_3d_state(bool flag);

            // Interface -- derived classes must implement these
            virtual Eigen::Vector4d GetBoundingBox2d() const = 0;
            virtual Eigen::Vector2d GetBoundingBox2dVelocity() const = 0;
            virtual Eigen::Vector3d GetSize3d() const = 0;
            virtual Eigen::Vector2d GetPoseGroundPlane() const = 0;
            virtual Eigen::Vector2d GetVelocityGroundPlane() const = 0;
            virtual Eigen::Matrix2d GetPoseCovariance() const = 0;
            virtual Eigen::Matrix3d GetSizeCovariance() const = 0;
            virtual Eigen::Matrix2d GetVelocityCovariance() const = 0;
            virtual Eigen::Matrix4d GetBoundingBox2dCovariance() const = 0;

            SUN::utils::Camera camera_current_frame_; // Needs to be updated every frame!
            bool use_2d_measurement_to_update_3d_state_; // Needs to be updated every frame!

            // Typedefs
            typedef std::shared_ptr<const CoupledExtendedKalmanFilter> ConstPtr;
            typedef std::shared_ptr<CoupledExtendedKalmanFilter> Ptr;
        };
    }
}

#endif // GOT_KALMAN_FILTER_H

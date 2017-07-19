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

#ifndef GOT_KALMAN_FILTER_CONST_VELOCITY_2D_3D_H
#define GOT_KALMAN_FILTER_CONST_VELOCITY_2D_3D_H

#include <tracking/extended_kalman_filter.h>

namespace GOT {
    namespace tracking {

        /**
           * @brief Implementation of the coupled 2D-3D state filter (ICRA'17).
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class CoupledImageWorldSpaceFilter : public CoupledExtendedKalmanFilter {
        public:
            struct Parameters : public KalmanFilter::Parameters {
                double dt; // Time between two measurements

                // These two are weighting factors for the 2D<->3D states filtering feedback
                double w2;
                double w3;

                bool states_coupling_;

                Parameters() : KalmanFilter::Parameters(), dt(0.1) {
                    w2 = 0.5;
                    w3 = (1.0 - w2);
                    states_coupling_ = true;
                }
            };

            CoupledImageWorldSpaceFilter(const Parameters &params);
            void Init(const Eigen::VectorXd &x_0, const Eigen::MatrixXd &P_0);

            Eigen::VectorXd NonlinearStateProjection() const;
            Eigen::MatrixXd LinearizeTransitionMatrix() const;
            Eigen::VectorXd compute_measurement_residual(const Eigen::VectorXd z_t, const Eigen::MatrixXd &H);

            // Interface
            Eigen::Vector4d GetBoundingBox2d() const;
            Eigen::Vector2d GetBoundingBox2dVelocity() const;
            Eigen::Vector3d GetSize3d() const;
            Eigen::Vector2d GetPoseGroundPlane() const;
            Eigen::Vector2d GetVelocityGroundPlane() const;
            Eigen::Matrix2d GetPoseCovariance() const;
            Eigen::Matrix3d GetSizeCovariance() const;
            Eigen::Matrix2d GetVelocityCovariance() const;
            Eigen::Matrix4d GetBoundingBox2dCovariance() const;

            Parameters params_;
        };
    }
}

#endif //GOT_KALMAN_FILTER_CONST_VELOCITY_2D_3D_H

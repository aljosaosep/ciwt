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

#include "coupled_image_world_space_filter.h"

// utils
#include "sun_utils/ground_model.h"

namespace GOT {
    namespace tracking {

        auto project_height = [](double h_3d_in, double focal_length, double dist_from_cam)->double {
            return h_3d_in * focal_length / dist_from_cam; // P
        };

        auto back_project_height = [](double h_2d_in, double focal_length, double dist_from_cam)->double {
            return h_2d_in * dist_from_cam / focal_length; // BP
        };

        // -------------------------------------------------------------------------------
        // +++ Implementation: coupled 2D-3D state filter +++
        // -------------------------------------------------------------------------------
        CoupledImageWorldSpaceFilter::CoupledImageWorldSpaceFilter(const Parameters &params)
                : params_(params), CoupledExtendedKalmanFilter(static_cast<KalmanFilter::Parameters>(params)) {

        }

        void CoupledImageWorldSpaceFilter::Init(const Eigen::VectorXd &x_0, const Eigen::MatrixXd &P_0) {
            // Set params
            this->state_vector_dim_ = params_.state_vector_dim_;
            const double dt = params_.dt; // Assume delta_t parameter is given

            assert(this->state_vector_dim_ == x_0.size());
            assert(this->state_vector_dim_ == P_0.rows());
            assert(this->state_vector_dim_ == P_0.cols());

            // Set initial state
            this->x_ = x_0;
            this->P_ = P_0;

            // Transition matrix
            A_.setIdentity(state_vector_dim_, state_vector_dim_);
            A_(0, 2) = dt;
            A_(1, 3) = dt;

            A_(7, 11) = 1;
            A_(8, 12) = 1;
            A_(9, 13) = 1;
            A_(10, 14) = 1;

            // Default system noise matrix
            G_.setIdentity(state_vector_dim_, state_vector_dim_);
            G_ *= 0.1 * 0.1; // Default process noise.
        }

        Eigen::Vector4d CoupledImageWorldSpaceFilter::GetBoundingBox2d() const {
            return this->x_.block(7, 0, 4, 1);
        }

        Eigen::Vector2d CoupledImageWorldSpaceFilter::GetBoundingBox2dVelocity() const {
            return this->x_.block(11, 0, 2, 1);
        }

        Eigen::Vector3d CoupledImageWorldSpaceFilter::GetSize3d() const {
            return this->x_.block(4, 0, 3, 1);
        }

        Eigen::Vector2d CoupledImageWorldSpaceFilter::GetPoseGroundPlane() const {
            return this->x_.head<2>();
        }

        Eigen::Vector2d CoupledImageWorldSpaceFilter::GetVelocityGroundPlane() const {
            return this->x_.block(2, 0, 2, 1);
        }

        Eigen::Matrix2d CoupledImageWorldSpaceFilter::GetPoseCovariance() const {
            return P_.block(0, 0, 2, 2);
        }

        Eigen::Matrix3d CoupledImageWorldSpaceFilter::GetSizeCovariance() const {
            return P_.block(4, 4, 3, 3);
        }

        Eigen::Matrix2d CoupledImageWorldSpaceFilter::GetVelocityCovariance() const {
            return P_.block(2, 2, 2, 2);
        }

        Eigen::Matrix4d CoupledImageWorldSpaceFilter::GetBoundingBox2dCovariance() const {
            return P_.block(7, 7, 4, 4);
        }

        Eigen::VectorXd CoupledImageWorldSpaceFilter::NonlinearStateProjection() const {
            Eigen::VectorXd x_prio = A_ * x_; // + u; // Linear part

            if (params_.states_coupling_) {
                // Need y-component of the filtered state ...
                const double x = x_[0];
                const double z = x_[1];
                const Eigen::VectorXd &plane_params_camera_space = std::static_pointer_cast<SUN::utils::PlanarGroundModel>(
                        camera_current_frame_.ground_model())->plane_params();
                double height_world = camera_current_frame_.ComputeGroundPlaneHeightInWorldSpace(x, z, plane_params_camera_space);

                // ... to project the point to cam. space and compute distance from camera.
                Eigen::Vector4d p_cam = camera_current_frame_.WorldToCamera(Eigen::Vector4d(x, height_world, z, 1.0));
                double dist_from_cam = p_cam[2];

                if (dist_from_cam < 0.1) {
                    return Eigen::VectorXd(); // Return empty
                }

                // Back-project 3D height estim. to image, project 2D height estim. to world using estimated 3D pose
                double new_h_2d = project_height(x_prio[5], camera_current_frame_.f_u(), dist_from_cam);
                double new_h_3d = back_project_height(x_prio[10], camera_current_frame_.f_u(), dist_from_cam);

                assert (!(std::isnan(new_h_3d) || std::isnan(new_h_2d)));

                // Update 'h3d'
                if (use_2d_measurement_to_update_3d_state_)
                    x_prio[5] = params_.w3 * x_prio[5] + params_.w2 * new_h_3d;

                // Update 'h2d'
                x_prio[10] = params_.w2 * x_prio[10] + params_.w3 * new_h_2d;

                // Compute predicted 2D bounding-box center
                Eigen::Vector3i projected_footpoint_image = camera_current_frame_.CameraToImage(p_cam);
                auto proj_cx = static_cast<double>(projected_footpoint_image[0]);
                auto proj_cy = static_cast<double>(projected_footpoint_image[1]) - x_prio[10] / 2.0;

                // Update 'cx' and 'cy'
                x_prio[7] = params_.w2 * x_prio[7] + params_.w3 * proj_cx;
                x_prio[8] = params_.w2 * x_prio[8] + params_.w3 * proj_cy;// TODO: take original h!
            }

            return x_prio;
        }

        Eigen::MatrixXd CoupledImageWorldSpaceFilter::LinearizeTransitionMatrix() const {

            // Need y-component of the filtered state ...
            const double x = x_[0];
            const double z = x_[1];
            const Eigen::VectorXd &plane_params_camera_space = std::static_pointer_cast<SUN::utils::PlanarGroundModel>(camera_current_frame_.ground_model())->plane_params();
            double height_world = camera_current_frame_.ComputeGroundPlaneHeightInWorldSpace(x, z, plane_params_camera_space);

            // ... to project the point to cam. space and compute distance from camera.
            Eigen::Vector4d p_cam = camera_current_frame_.WorldToCamera(Eigen::Vector4d(x, height_world, z, 1.0));
            double dist_from_cam = p_cam[2];

            // Update the transition matrix
            Eigen::MatrixXd A_linearized = this->A_;

            if (params_.states_coupling_) {
                A_linearized(8, 1) = params_.w3 / dist_from_cam;
                A_linearized(8, 3) = (this->params_.dt * params_.w3) / dist_from_cam;
                A_linearized(5, 5) = params_.w2;
                A_linearized(10, 5) = (camera_current_frame_.f_u() * params_.w3) / dist_from_cam;
                A_linearized(8, 8) = params_.w2;
                A_linearized(5, 10) = (dist_from_cam * params_.w3) / camera_current_frame_.f_v();
                A_linearized(10, 10) = params_.w2;
                A_linearized(8, 12) = params_.w2;
                A_linearized(10, 14) = params_.w2;
            }

            return A_linearized;
        }

        Eigen::VectorXd CoupledImageWorldSpaceFilter::compute_measurement_residual(const Eigen::VectorXd z_t, const Eigen::MatrixXd &H) {
            return z_t - H*x_;
        }
    }
}
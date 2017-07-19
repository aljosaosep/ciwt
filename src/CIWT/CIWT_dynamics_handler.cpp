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

#include "CIWT_dynamics_handler.h"

// utils
#include "sun_utils/ground_model.h"

// CIWT
#include "coupled_image_world_space_filter.h"

namespace GOT {
    namespace tracking {

        auto convert_bounding_box_2d_top_left_corner_to_top_mid_corner = [](const Eigen::Vector4d &bounding_box_2d)->Eigen::Vector4d {
            Eigen::Vector4d reparametrized_bbox;
            reparametrized_bbox << bounding_box_2d[0]+bounding_box_2d[2]/2,
                    bounding_box_2d[1], bounding_box_2d[2], bounding_box_2d[3];
            return reparametrized_bbox;
        };

        auto convert_bounding_box_2d_top_left_corner_to_center_mid_corner = [](const Eigen::Vector4d &bounding_box_2d)->Eigen::Vector4d {
            Eigen::Vector4d reparametrized_bbox;
            reparametrized_bbox << bounding_box_2d[0]+bounding_box_2d[2]/2,
                    bounding_box_2d[1]+bounding_box_2d[3]/2, bounding_box_2d[2], bounding_box_2d[3];
            return reparametrized_bbox;
        };


        // -------------------------------------------------------------------------------
        // +++ CIWT dynamics handler implementation +++
        // -------------------------------------------------------------------------------
        void DynamicsHandlerCIWT::InitializeState(const SUN::utils::Camera &camera, const Observation &obs, bool forward, Hypothesis &hypo) {

            // Create filter object
            CoupledImageWorldSpaceFilter::Parameters params;
            params.dt = this->parameters_["dt"].as<double>();
            params.state_vector_dim_ = this->state_dim_;
            params.store_history_ = true;
            params.states_coupling_ = this->enable_kalman_coupling_;
            hypo.kalman_filter() = std::shared_ptr<CoupledImageWorldSpaceFilter>(new CoupledImageWorldSpaceFilter(params));

            Eigen::Vector4d reparam_bbox_2d = convert_bounding_box_2d_top_left_corner_to_center_mid_corner(obs.bounding_box_2d());
            //Eigen::Vector4d reparam_bbox_2d = convert_bounding_box_2d_top_left_corner_to_top_mid_corner(obs.bounding_box_2d());

            // Ground-plane pose and velocity. Perform projection camera->world, take into account time arrow!
            Eigen::Vector4d footpoint_world = camera.CameraToWorld(obs.footpoint());
            Eigen::MatrixXd pose_cov_world = camera.R() * obs.covariance3d() * camera.R().transpose();

            // Physical size
            double w_3d = obs.bounding_box_3d()[3];
            double h_3d = obs.bounding_box_3d()[4];
            double d_3d = obs.bounding_box_3d()[5];

            // Full 3D filter?

            bool got_velocity = !(std::isnan(obs.velocity()[0]));
            bool use_3d_obs_process = (obs.footpoint()[2] < this->max_dist_use_stereo_) && obs.proposal_3d_avalible() && got_velocity;

            if (use_3d_obs_process) {

                /// Initalize using 3D segmentation info

                // Velocity
                Eigen::Vector3d velocity_world;
                if (forward) velocity_world = camera.R() * obs.velocity();
                else velocity_world = camera.R() * (-1.0*obs.velocity());

                // Init state
                // [x,y,dx,dy,w,h,d | x_b2d, y_b2d, w_b2d, h_b2d, dx_b2d, dy_b2d]
                Eigen::VectorXd kalman_init_state(params.state_vector_dim_);
                kalman_init_state.setZero();
                kalman_init_state << footpoint_world[0], footpoint_world[2],
                        velocity_world[0], velocity_world[2],
                        w_3d, h_3d, d_3d,
                        reparam_bbox_2d,
                        0.0, 0.0,
                        0.0, 0.0;

                Eigen::MatrixXd P0_measured;
                P0_measured.setZero(params.state_vector_dim_,params.state_vector_dim_);
                P0_measured(0, 0) = pose_cov_world(0, 0);
                P0_measured(0, 1) = pose_cov_world(0, 2);
                P0_measured(1, 0) = pose_cov_world(2, 0);
                P0_measured(1, 1) = pose_cov_world(2, 2);

                const Eigen::Matrix2d &velocity_obs_cov = P0_measured.block(0, 0, 2, 2);
                P0_measured.block(2, 2, 2, 2) = 2.0*velocity_obs_cov; // Approximation!

                // Size uncertainty
                P0_measured(4, 4) = 0.2*0.2;
                P0_measured(5, 5) = 0.2*0.2;
                P0_measured(6, 6) = 0.5*0.5;
                P0_measured.block(7,7,8, 8) = bounding_box_initial_variance_*bounding_box_initial_variance_*Eigen::MatrixXd::Identity(8, 8);

                hypo.kalman_filter()->Init(kalman_init_state, P0_measured);
                hypo.kalman_filter()->set_G(G_car_);
            }
            else {
                /// Initalize using 3D segmentation info not avalible
                /// => use category priors where applicable, init velocity with large uncertainty
                Eigen::VectorXd kalman_init_state;
                kalman_init_state.setZero(params.state_vector_dim_);

                // Select appropriate transition matrices ...
                Eigen::MatrixXd P_init;
                Eigen::MatrixXd G_init;
                if (obs.detection_category()==SUN::shared_types::PEDESTRIAN) { //TODO We need a DOG class.
                    P_init = P0_pedestrian_;
                    G_init = G_pedestrian_;
                }
                else if (obs.detection_category()==SUN::shared_types::CAR){
                    P_init = P0_car_;
                    G_init = G_car_;
                }
                else if (obs.detection_category()==SUN::shared_types::CYCLIST) {
                    P_init = P0_car_;
                    G_init = G_car_;
                }
                else {
                    std::cout << "ERROR: category not known; for generic objects, you should always receive depth measurements! No good! Fix! " << std::endl;
                }

                kalman_init_state << footpoint_world[0], footpoint_world[2],
                        1e-5, 1e-5, // TODO setting velocity to 0 maybe not the best idea?!?
                        w_3d, h_3d, d_3d,
                        reparam_bbox_2d,
                        0.0, 0.0,
                        0.0, 0.0;

                ///  --- NEW 2017_07_11: override pos. uncertainty ---
                P_init(0, 0) = pose_cov_world(0, 0);
                P_init(0, 1) = pose_cov_world(0, 2);
                P_init(1, 0) = pose_cov_world(2, 0);
                P_init(1, 1) = pose_cov_world(2, 2);
                /// --- NEW ---

                hypo.kalman_filter()->Init(kalman_init_state, P_init);
                hypo.kalman_filter()->set_G(G_init);
            }
        }

        void DynamicsHandlerCIWT::ApplyTransition(const SUN::utils::Camera &camera, const Eigen::VectorXd &u, Hypothesis &hypo) {
            assert(u.size()==hypo.kalman_filter_const()->x().size());

            //bool is_hypo_exiting = hypo.is_about_to_leave_the_frustum();

            // If hypo is leaving the view. frustum, be careful how to handle the 2d bbox
            Eigen::MatrixXd G_orig = hypo.kalman_filter()->G();

            /// 'Normal' transition model.
            hypo.kalman_filter()->set_camera_current_frame(camera);
            //hypo.kalman_filter()->set_use_2d_measurement_to_update_3d_state(!is_hypo_exiting);
            hypo.kalman_filter()->Prediction(u);

            Eigen::VectorXd kf_state_after_projection = hypo.kalman_filter_const()->x();

            // Our bounding-box 2d physical size can 'drift' below 0. Let's not allow that and enforce minimal bbox size!
            if (kf_state_after_projection[9] <= 10)
                kf_state_after_projection[9] = 10;

            if (kf_state_after_projection[10] <= 10)
                kf_state_after_projection[10] = 10;

//            // Proj. velocity limit
//            if (kf_state_after_projection[11] >= 10)
//                kf_state_after_projection[11] = 10;
//
//            if (kf_state_after_projection[12] >= 10)
//                kf_state_after_projection[12] = 10;

            // 'reset' bbox size change rate
            kf_state_after_projection[13] = 0.0;
            kf_state_after_projection[14] = 0.0;
            hypo.kalman_filter()->set_x(kf_state_after_projection);
        }

        void DynamicsHandlerCIWT::ApplyCorrection(const SUN::utils::Camera &camera, const Observation &obs, bool forward, Hypothesis &hypo) {

            Eigen::Vector4d reparam_bbox_2d = convert_bounding_box_2d_top_left_corner_to_center_mid_corner(obs.bounding_box_2d());
            //Eigen::Vector4d reparam_bbox_2d = convert_bounding_box_2d_top_left_corner_to_top_mid_corner(obs.bounding_box_2d());

            // Pose and its uncertainty (on the ground plane, hence, consider only X-Z components).
            Eigen::Vector4d footpoint_world = camera.CameraToWorld(obs.footpoint());
            Eigen::MatrixXd pose_cov_world = camera.R() * obs.covariance3d() * camera.R().transpose();

            // Physical size.
            double w_3d = obs.bounding_box_3d()[3];
            double h_3d = obs.bounding_box_3d()[4];
            double d_3d = obs.bounding_box_3d()[5];

            Eigen::MatrixXd R; // Observation covariance matrix.
            Eigen::VectorXd measurement; // Observation vector.

            // This param. specifies up to what distance 'full 3D' observation model is used.

            bool got_velocity = !(std::isnan(obs.velocity()[0]));
            bool use_3d_obs_process = (obs.footpoint()[2] < this->max_dist_use_stereo_) && obs.proposal_3d_avalible() && got_velocity;

            if (use_3d_obs_process) {
                /// '3D' observation model
                /// Here, we measure the full 3D state and 2D bounding box.

                // Velocity
                Eigen::Vector3d velocity_world;
                if (forward) velocity_world = camera.R() * obs.velocity();
                else velocity_world = camera.R() * (-1.0*obs.velocity());

                // Pose unc.
                R.setZero(11, 11);
                R(0, 0) = pose_cov_world(0, 0);
                R(0, 1) = pose_cov_world(0, 2);
                R(1, 0) = pose_cov_world(2, 0);
                R(1, 1) = pose_cov_world(2, 2);

                // Velocity unc.
                Eigen::Matrix2d velocity_obs_cov = 2.0 * R.block(0, 0, 2, 2); // This is approx. correct. The really correct would be to sum-up cov's from two point measurements.
                R.block(2, 2, 2, 2) = velocity_obs_cov;

                // 3D size unc.
                R(4, 4) = 0.2*0.2;
                R(5, 5) = 0.2*0.2;
                R(6, 6) = 0.5*0.5;

                // 2D bounding box unc.
                R.block(7,7,4,4) = Eigen::MatrixXd::Identity(4,4)*bounding_box_obs_noise_*bounding_box_obs_noise_;

                measurement.setZero(11);
                measurement << footpoint_world[0], footpoint_world[2],
                        velocity_world[0], velocity_world[2],
                        w_3d, h_3d, d_3d,
                        reparam_bbox_2d;

                Eigen::MatrixXd H_tmp;
                H_tmp.setZero(11,15);
                H_tmp.block(0,0,11,11) = Eigen::MatrixXd::Identity(11, 11);
                hypo.kalman_filter()->Correction(measurement, R, H_tmp);
            } else {
                /// '2D' observation model
                /// Here, we measure the full 3D position and size, and 2D bounding box.

                // Measured position uncertainty
                R.setZero(9,9);
                R(0, 0) = pose_cov_world(0, 0);
                R(0, 1) = pose_cov_world(0, 2);
                R(1, 0) = pose_cov_world(2, 0);
                R(1, 1) = pose_cov_world(2, 2);

                // 3D size variance
                R(2,2) = 0.5*0.5;
                R(3,3) = 0.5*0.5;
                R(4,4) = 0.5*0.5;

                R.block(5,5,4,4) = Eigen::MatrixXd::Identity(4,4)*bounding_box_obs_noise_*bounding_box_obs_noise_;
                measurement.setZero(9);
                measurement << footpoint_world[0], footpoint_world[2], w_3d, h_3d, d_3d, reparam_bbox_2d; // bbox_cx, bbox_cy, bbox_w, bbox_h;
                hypo.kalman_filter()->Correction(measurement, R, H_observe_pos_bbox2d_size_);
            }
        }

        void DynamicsHandlerCIWT::InitializeMatrices() {

            /// Initial uncertainty: Pedestrian
            P0_pedestrian_.setZero(state_dim_,state_dim_);
            P0_pedestrian_(0, 0) = 0.4; // Pose
            P0_pedestrian_(1, 1) = 1.2;
            P0_pedestrian_(2, 2) = P0_ped_velocity_variance_[0]; // Velocity
            P0_pedestrian_(3, 3) = P0_ped_velocity_variance_[1];
            P0_pedestrian_(4, 4) = 0.14*0.14;
            P0_pedestrian_(5, 5) = 0.1*0.1;
            P0_pedestrian_(6, 6) = 0.2*0.2;
            P0_pedestrian_.block(7,7,8, 8) = bounding_box_initial_variance_*bounding_box_initial_variance_*Eigen::MatrixXd::Identity(8, 8);

            /// Initial uncertainty: Car
            P0_car_.setZero(state_dim_,state_dim_);
            P0_car_(0, 0) = 1.0; // Pose
            P0_car_(1, 1) = 1.0;
            P0_car_(2, 2) = P0_car_velocity_variance_[0]; // Velocity
            P0_car_(3, 3) = P0_car_velocity_variance_[1];
            P0_car_(4, 4) = 0.14*0.14;
            P0_car_(5, 5) = 0.253*0.253;
            P0_car_(6, 6) = 0.57*0.57;
            P0_car_.block(7,7,8, 8) = bounding_box_initial_variance_*bounding_box_initial_variance_*Eigen::MatrixXd::Identity(8, 8);

            /// System noise: Pedestrian
            G_pedestrian_.setZero(state_dim_,state_dim_);
            G_pedestrian_(0, 0) = G_ped_pos_variance_[0]*G_ped_pos_variance_[0];
            G_pedestrian_(1, 1) = G_ped_pos_variance_[1]*G_ped_pos_variance_[1];
            G_pedestrian_(2, 2) = G_ped_velocity_variance_[0]*G_ped_velocity_variance_[0];
            G_pedestrian_(3, 3) = G_ped_velocity_variance_[1]*G_ped_velocity_variance_[1];
            G_pedestrian_(4, 4) = 0.05 * 0.05; // w
            G_pedestrian_(5, 5) = 0.05 * 0.05; // h
            G_pedestrian_(6, 6) = 0.05 * 0.05; // l
            Eigen::MatrixXd bbox_system_noise_matrix;
            bbox_system_noise_matrix.setIdentity(8,8);
            bbox_system_noise_matrix *= bounding_box_system_noise_*bounding_box_system_noise_;
            G_pedestrian_.block(7,7, 8, 8) = bbox_system_noise_matrix;

            /// System noise: Car
            G_car_.setZero(state_dim_,state_dim_);
            G_car_(0, 0) = G_car_pos_variance_[0]*G_car_pos_variance_[0];
            G_car_(1, 1) = G_car_pos_variance_[1]*G_car_pos_variance_[1];
            G_car_(2, 2) = G_car_velocity_variance_[0] * G_car_velocity_variance_[0];
            G_car_(3, 3) = G_car_velocity_variance_[1] * G_car_velocity_variance_[1];
            G_car_(4, 4) = 0.05 * 0.05; // w
            G_car_(5, 5) = 0.05 * 0.05; // h
            G_car_(6, 6) = 0.05 * 0.05; // l
            G_car_.block(7,7, 8, 8) = bbox_system_noise_matrix;

            /// Partial observation
            H_observe_pos_bbox2d_size_.setZero(9, 15);
            H_observe_pos_bbox2d_size_.block(0,0,2,2) = Eigen::Matrix2d::Identity();
            H_observe_pos_bbox2d_size_.block(2,4,7,7) = Eigen::MatrixXd::Identity(7,7);

            /// Full-observation
            H_observe_everything_.setZero(13,15);
            H_observe_everything_.block(0,0,13,13) = Eigen::MatrixXd::Identity(13,13);
        }

        DynamicsHandlerCIWT::DynamicsHandlerCIWT(const po::variables_map &params) : DynamicsModelHandler(params) {
            state_dim_ = 15;
            enable_kalman_coupling_ = false;
            max_dist_use_stereo_ = 20.0;

            double kf_2d_obs = params.at("kf_2d_observation_noise").as<double>();
            double kf_2d_init = params.at("kf_2d_initial_variance").as<double>();
            double kf_2d_sys = params.at("kf_2d_system_noise").as<double>();

            double kf_init_vel_x = params.at("kf_init_velocity_variance_car_x").as<double>();
            double kf_init_vel_z = params.at("kf_init_velocity_variance_car_z").as<double>();

            double kf_sys_pos_x = params.at("kf_system_pos_variance_car_x").as<double>();
            double kf_sys_pos_z = params.at("kf_system_pos_variance_car_z").as<double>();

            double kf_sys_vel_x = params.at("kf_system_velocity_variance_car_x").as<double>();
            double kf_sys_vel_z = params.at("kf_system_velocity_variance_car_z").as<double>();

            bounding_box_obs_noise_ = kf_2d_obs;
            bounding_box_system_noise_ = kf_2d_sys;
            bounding_box_initial_variance_ = kf_2d_init;

            P0_car_velocity_variance_ = Eigen::Vector2d (kf_init_vel_x, kf_init_vel_z);
            P0_ped_velocity_variance_ = Eigen::Vector2d (0.2, 0.2);

            G_ped_pos_variance_ = Eigen::Vector2d (0.05, 0.1);
            G_ped_velocity_variance_ = Eigen::Vector2d (0.1, 0.1);

            G_car_pos_variance_ = Eigen::Vector2d (kf_sys_pos_x, kf_sys_pos_z);
            G_car_velocity_variance_ = Eigen::Vector2d (kf_sys_vel_x, kf_sys_vel_z);

            this->InitializeMatrices();
        }
    }
}

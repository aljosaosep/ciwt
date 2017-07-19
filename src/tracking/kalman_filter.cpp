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

#include <tracking/kalman_filter.h>

// eigen
#include <Eigen/LU>

// std
#include <iostream>
#include <fstream>

namespace GOT {
    namespace tracking {

        // -------------------------------------------------------------------------------
        // +++ Implementation: Kalman Filter Base +++
        // -------------------------------------------------------------------------------

        KalmanFilter::KalmanFilter(const Parameters &params) : state_vector_dim_(0), params_(params)  {

        }

        void KalmanFilter::ComputePrediction(const Eigen::VectorXd &u, Eigen::VectorXd &x_prio,
                                             Eigen::MatrixXd &P_prio) const {
            x_prio = A_*x_ + u; // Linear

            // Project covariance ahead
            P_prio = A_*P_*A_.transpose() + G_ ;
        }

        void KalmanFilter::Prediction() {
            assert(this->state_vector_dim_==A_.rows());
            assert(this->state_vector_dim_==A_.cols());
            assert(this->state_vector_dim_==G_.rows());
            assert(this->state_vector_dim_==G_.cols());

            Eigen::VectorXd u_zero, x_prio;
            Eigen::MatrixXd P_prio;
            u_zero.setZero(x_.size());
            this->ComputePrediction(u_zero, x_prio, P_prio);

            if (x_prio.size() != this->params_.state_vector_dim_) {
                return;
            }

            x_ = x_prio;
            P_ = P_prio;

            // Optionally, store whole history. Eg. for visualization.
            if (params_.store_history_) {
                this->state_predictions_.push_back(x_);
                this->cov_predictions_.push_back(P_);
            }
        }

        void KalmanFilter::Prediction(const Eigen::VectorXd &u) {
            assert(this->state_vector_dim_==A_.rows());
            assert(this->state_vector_dim_==A_.cols());
            assert(this->state_vector_dim_==G_.rows());
            assert(this->state_vector_dim_==G_.cols());

            Eigen::VectorXd x_prio;
            Eigen::MatrixXd P_prio;
            this->ComputePrediction(u, x_prio, P_prio);

            if (x_prio.size() != this->params_.state_vector_dim_) {
                return;
            }

            x_ = x_prio;
            P_ = P_prio;

            // Optionally, store whole history. Eg. for visualization.
            if (params_.store_history_) {
                this->state_predictions_.push_back(x_);
                this->cov_predictions_.push_back(P_);
            }
        }

        /// H is the observation model matrix (residual = z_t - H*x)
        void KalmanFilter::Correction(const Eigen::VectorXd z_t, const Eigen::MatrixXd &observation_cov_t, const Eigen::MatrixXd &H) {
            Eigen::MatrixXd H_transposed = H.transpose();

            // Compute Kalman gain
            Eigen::MatrixXd HPH_plus_obs_cov= (H*P_*H_transposed)+observation_cov_t;
            Eigen::MatrixXd K = P_*H_transposed*HPH_plus_obs_cov.inverse();

            // Update the estimate
            Eigen::VectorXd measurement_residual = z_t - H*x_;

            x_ = x_ + K*measurement_residual; //(z_t - H_*x_);

            // Update the covariance
            Eigen::MatrixXd  KH = K*H;
            P_ = (Eigen::MatrixXd::Identity(KH.rows(), KH.cols()) - KH)*P_;

            // Optionally, store whole history. Eg. for visualization.
            if (params_.store_history_) {
                this->state_corrections_.push_back(x_);
                this->cov_corrections_.push_back(P_);
                this->measurements_.push_back(z_t);
                this->cov_measurements_.push_back(observation_cov_t);
                this->kalman_gain_.push_back(K);
            }
        }

        /// Format: /path/to/dir/name_%s.txt
        /// Note: state dim. is not stored. At the client side, it is assumed to be simply 'known'.
        void KalmanFilter::SaveHistoryToFile(const char *filename) const {
            if (!this->params_.store_history_) {
                std::cout << "KalmanFilter::Error: Attempting to save Kalman history, but the params.save_history flag is off! Abort!" << std::endl;
                return;
            }

            /// States
            char buff[500];
            snprintf(buff, 500, filename, "state_predictions");
            std::ofstream state_pred_stream(buff);

            snprintf(buff, 500, filename, "state_corrections");
            std::ofstream state_corr_stream(buff);

            snprintf(buff, 500, filename, "state_measurements");
            std::ofstream state_meas_stream(buff);

            for (const auto &vec : state_predictions_)
                state_pred_stream << vec.transpose() << std::endl;

            for (const auto &vec : state_corrections_)
                state_corr_stream << vec.transpose() << std::endl;

            for (const auto &vec : measurements_)
                state_meas_stream << vec.transpose() << std::endl;

            //! Covariances
            snprintf(buff, 500, filename, "covariance_predictions");
            std::ofstream cov_pred_stream(buff);

            snprintf(buff, 500, filename, "covariance_corrections");
            std::ofstream cov_corr_stream(buff);

            snprintf(buff, 500, filename, "covariance_measurements");
            std::ofstream cov_meas_stream(buff);

            for (const auto &mat : cov_predictions_)
                cov_pred_stream << mat << std::endl;

            for (const auto &mat : cov_corrections_)
                cov_corr_stream << mat << std::endl;

            for (const auto &mat : cov_measurements_)
                cov_meas_stream << mat << std::endl;

            // Close file streams
            state_pred_stream.close();
            state_corr_stream.close();
            state_meas_stream.close();

            cov_pred_stream.close();
            cov_corr_stream.close();
            cov_meas_stream.close();
        }

        // Setters / Getters
        const Eigen::VectorXd& KalmanFilter::x() const {
            return x_;
        }

        void KalmanFilter::set_x(const Eigen::VectorXd &state) {
            this->x_ = state;
        }

        const Eigen::MatrixXd& KalmanFilter::P() const {
            return this->P_;
        }

        void KalmanFilter::set_G(const Eigen::MatrixXd& G) {
            this->G_ = G;
        }

        const Eigen::MatrixXd& KalmanFilter::G() const {
            return this->G_;
        }

        const std::vector<Eigen::VectorXd>& KalmanFilter::state_predictions() const {
            return this->state_predictions_;
        }

        const std::vector<Eigen::VectorXd>& KalmanFilter::state_corrections() const {
            return this->state_corrections_;
        }

        const std::vector<Eigen::MatrixXd>& KalmanFilter::cov_predictions() const {
            return this->cov_predictions_;
        }

        const std::vector<Eigen::MatrixXd>& KalmanFilter::cov_corrections() const {
            return this->cov_corrections_;
        }

        const std::vector<Eigen::VectorXd>& KalmanFilter::measurements() const {
            return this->measurements_;
        }

        const std::vector<Eigen::MatrixXd>& KalmanFilter::cov_measurements() const {
            return this->cov_measurements_;
        }

        const KalmanFilter::Parameters KalmanFilter::parameters() const {
            return params_;
        }

        void KalmanFilter::set_A(const Eigen::MatrixXd &A) {
            this->A_ = A;
        }

        const Eigen::MatrixXd &KalmanFilter::A() const {
            return this->A_;
        }

        const std::vector<Eigen::MatrixXd> &KalmanFilter::kalman_gains() const {
            return this->kalman_gain_;
        }

        // -------------------------------------------------------------------------------
        // +++ Implementation: Little example: simple Const-Velocity model +++
        // -------------------------------------------------------------------------------

        ConstantVelocityKalmanFilter::ConstantVelocityKalmanFilter(const Parameters &params) : params_(params), KalmanFilter(
                static_cast<KalmanFilter::Parameters>(params)) {
        }

        void ConstantVelocityKalmanFilter::Init(const Eigen::VectorXd &x_0, const Eigen::MatrixXd &P_0) {
            // Set params
            this->state_vector_dim_ = params_.state_vector_dim_;
            const double dt = params_.dt; // Assume delta_t parameter is given

            assert(this->state_vector_dim_==x_0.size());
            assert(this->state_vector_dim_==P_0.rows());
            assert(this->state_vector_dim_==P_0.cols());

            // Set initial state
            this->x_ = x_0;
            this->P_ = P_0;

            // Set up transition matrix A (simply adds change in velocity to pos.)
            A_.setIdentity(state_vector_dim_, state_vector_dim_);
            A_(0,2) = dt;
            A_(1,3) = dt;

            G_ = Eigen::Matrix4d::Identity() *0.1*0.1; // Default process noise.
        }
    }
}

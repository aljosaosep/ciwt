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

#ifndef GOT_KALMAN_FILTER_H
#define GOT_KALMAN_FILTER_H

// std
#include <memory>

// Eigen
#include <Eigen/Core>

// Forward decl.
#include "sun_utils/camera.h"

namespace GOT {
    namespace tracking {

        /**
           * @brief Base class for Kalman-filtering. Just an interface.
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class KalmanFilter {
        public:
            struct Parameters {
                bool store_history_;
                int state_vector_dim_;
                Parameters() : store_history_(false), state_vector_dim_(0) { } // Default params
            };

            KalmanFilter(const Parameters &params);

            /**
               * @brief This method should set the initial state and set-up the fixed matrices.
               */
            virtual void Init(const Eigen::VectorXd &x_0, const Eigen::MatrixXd &P_0)=0;
            virtual void ComputePrediction(const Eigen::VectorXd &u, Eigen::VectorXd &x_prio, Eigen::MatrixXd &P_prio) const;
            virtual void Prediction();
            virtual void Prediction(const Eigen::VectorXd &u);

            /**
               * @brief Use measurement to correct the state.
               * @param z_t Current measurement.
               * @param observation_cov_t Current observation covariance matrix.
               */
            virtual void Correction(const Eigen::VectorXd z_t, const Eigen::MatrixXd &observation_cov_t, const Eigen::MatrixXd &H);

            // Setters / Getters
            const Eigen::VectorXd& x() const;
            const Eigen::MatrixXd& P() const;
            const Eigen::MatrixXd& A() const;
            const Eigen::MatrixXd& G() const;

            void set_G(const Eigen::MatrixXd& G);
            void set_A(const Eigen::MatrixXd& A);
            void set_x(const Eigen::VectorXd &state);

            const std::vector<Eigen::VectorXd> &state_predictions() const;
            const std::vector<Eigen::VectorXd> &state_corrections() const;
            const std::vector<Eigen::MatrixXd> &cov_predictions() const;
            const std::vector<Eigen::MatrixXd> &cov_corrections() const;
            const std::vector<Eigen::VectorXd> &measurements() const;
            const std::vector<Eigen::MatrixXd> &cov_measurements() const;
            const std::vector<Eigen::MatrixXd> &kalman_gains() const;


            const Parameters parameters() const;

            void SaveHistoryToFile(const char *filename) const;

            // Typedefs
            typedef std::shared_ptr<const KalmanFilter> ConstPtr;
            typedef std::shared_ptr<KalmanFilter> Ptr;

        protected:
            Eigen::VectorXd x_; // State vector
            Eigen::MatrixXd P_; // State covariance

            Eigen::MatrixXd A_; // Transition model
            Eigen::MatrixXd B_; // Control-input model

            Eigen::MatrixXd G_; // State cov. noise

            int state_vector_dim_;

            std::vector<Eigen::VectorXd> state_predictions_;
            std::vector<Eigen::VectorXd> state_corrections_;
            std::vector<Eigen::MatrixXd> cov_predictions_;
            std::vector<Eigen::MatrixXd> cov_corrections_;
            std::vector<Eigen::VectorXd> measurements_;
            std::vector<Eigen::MatrixXd> cov_measurements_;
            std::vector<Eigen::MatrixXd> kalman_gain_;

            Parameters params_;
        };

        /**
          * @brief Implementation of const-velocity Kalman filter.
          * @author Aljosa (osep@vision.rwth-aachen.de)
          */
        class ConstantVelocityKalmanFilter : public KalmanFilter {
        public:
            struct Parameters : public KalmanFilter::Parameters {
                double dt; // Time between two measurements

                Parameters() : KalmanFilter::Parameters(), dt(0.0) { }
            };

            ConstantVelocityKalmanFilter(const Parameters &params);
            void Init(const Eigen::VectorXd &x_0, const Eigen::MatrixXd &P_0);

            Parameters params_;
        };

    }
}

#endif // GOT_KALMAN_FILTER_H

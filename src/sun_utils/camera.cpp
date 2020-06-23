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

#include "camera.h"

// eigen
#include <Eigen/LU>

namespace SUN {
    namespace utils {

        Camera::Camera() {
            this->P_.setIdentity();
            this->Rt_.setIdentity();
            this->Rt_inv_.setIdentity();
            this->K_.setIdentity();
            this->K_inv_.setIdentity();
            reference_pose_.setZero();
        }

        bool Camera::ComputeMeasurementCovariance3d(const Eigen::Vector3d &pos_3d, Eigen::Matrix2d c2d_mat, const Eigen::Matrix<double, 3, 4>& P_left_cam, const Eigen::Matrix<double, 3, 4>& P_right_cam,  Eigen::Matrix3d &covariance_out) {
            //! Compute inverse of pixel-measurement covariance matrix.
            auto inv_tmp = c2d_mat.inverse().eval();
            c2d_mat = inv_tmp;

            //! Compute Jacobians of projection matrices of both cameras.
            double wl = P_left_cam(2,0)*pos_3d[0] + P_left_cam(2,1)*pos_3d[1] + P_left_cam(2,2)*pos_3d[2] + P_left_cam(2,3);
            double wl2 = wl * wl;

            double wr = P_right_cam(2,0)*pos_3d[0] + P_right_cam(2,1)*pos_3d[1] + P_right_cam(2,2)*pos_3d[2] + P_right_cam(2,3);
            double wr2 = wr * wr;

            Eigen::Matrix<double, 2, 3> FL;
            FL.setZero();

            Eigen::Matrix<double, 2, 3> FR;
            FR.setZero();

            for (int i = 0; i < 2; i++) {
                FL(i,0) = P_left_cam(i,0) / wl - (P_left_cam(i,0) * pos_3d[0] + P_left_cam(i,1) * pos_3d[1] + P_left_cam(i,2) * pos_3d[2] + P_left_cam(i,3)) * P_left_cam(2,0) / wl2;
                FL(i,1) = P_left_cam(i,1) / wl - (P_left_cam(i,0) * pos_3d[0] + P_left_cam(i,1) * pos_3d[1] + P_left_cam(i,2) * pos_3d[2] + P_left_cam(i,3)) * P_left_cam(2,1) / wl2;
                FL(i,2) = P_left_cam(i,2) / wl - (P_left_cam(i,0) * pos_3d[0] + P_left_cam(i,1) * pos_3d[1] + P_left_cam(i,2) * pos_3d[2] + P_left_cam(i,3)) * P_left_cam(2,2) / wl2;

                FR(i,0) = P_right_cam(i,0) / wr - (P_right_cam(i,0) * pos_3d[0] + P_right_cam(i,1)  * pos_3d[1] + P_right_cam(i,2) * pos_3d[2] + P_right_cam(i,3)) * P_right_cam(2,0) / wr2;
                FR(i,1) = P_right_cam(i,1) / wr - (P_right_cam(i,0) * pos_3d[0] + P_right_cam(i,1)  * pos_3d[1] + P_right_cam(i,2) * pos_3d[2] + P_right_cam(i,3)) * P_right_cam(2,1) / wr2;
                FR(i,2) = P_right_cam(i,2) / wr - (P_right_cam(i,0) * pos_3d[0] + P_right_cam(i,1)  * pos_3d[1] + P_right_cam(i,2) * pos_3d[2] + P_right_cam(i,3)) * P_right_cam(2,2) / wr2;
            }

            //! Sum left and right part and compute the inverse.
            Eigen::Matrix3d FLTFL = (FL.transpose()*c2d_mat*FL);
            Eigen::Matrix3d FRTFR = (FR.transpose()*c2d_mat*FR);
            Eigen::Matrix3d sum = FLTFL+FRTFR;

            // Make sure pose cov matrix is well-conditioned
            Eigen::Matrix3d gaussian_prior;
            const double prior_fact = 0.05;
            gaussian_prior << prior_fact, 0.0, 0.0, 0.0, prior_fact, 0.0, 0.0, 0.0, prior_fact;
            sum += gaussian_prior;

            bool invertible;
            double determinant;
            sum.computeInverseAndDetWithCheck(covariance_out, determinant, invertible);
            return invertible;
        }

        bool Camera::ComputeMeasurementCovariance3d(const Eigen::Vector3d &pos_3d, double c_2d, const Eigen::Matrix<double, 3, 4>& P_left_cam, const Eigen::Matrix<double, 3, 4>& P_right_cam,  Eigen::Matrix3d &covariance_out) {
            // Compute pixel cov. mat.
            Eigen::Matrix2d c2d_mat;
            c2d_mat.setIdentity();
            c2d_mat *= c_2d;

            bool invertible = ComputeMeasurementCovariance3d(pos_3d, c2d_mat, P_left_cam, P_right_cam, covariance_out);
            assert(invertible);
            return invertible;
        }

        void Camera::init(const Eigen::Matrix<double,3,4> &reference_projection_matrix, const Eigen::Matrix4d &odometry_matrix, int width, int height) {
            /// Set up K, Rt matrices and their inverses
            this->K_ = reference_projection_matrix.block<3,3>(0,0);
            this->K_inv_ = K_.inverse();

            /// Little kitti hack, set t part to 0
            /// because t is specified relative to camera 0
            // TODO (Aljosa) sort this out!
            auto ref_proj_mat_copy = reference_projection_matrix;
            ref_proj_mat_copy.col(3).head<3>() = Eigen::Vector3d(0.0,0.0,0.0);

            /// Compute reference pose (pose in the first frame)
            reference_pose_(2,0) = ref_proj_mat_copy(2,3);
            reference_pose_(0,0) = (ref_proj_mat_copy(0,3) - ref_proj_mat_copy(0,2)*reference_pose_(2,0)) / ref_proj_mat_copy(0,0);
            reference_pose_(1,0) = (ref_proj_mat_copy(1,3) - ref_proj_mat_copy(1,2)*reference_pose_(2,0)) / ref_proj_mat_copy(1,1);
            reference_pose_(3,0) = 1.0;

            this->ApplyPoseTransform(odometry_matrix);

            this->P_.setIdentity();
            this->P_.block(0,0,3,4) = reference_projection_matrix;

            /// width, height
            width_ = width;
            height_ = height;
        }

        void Camera::init(const Eigen::Matrix3d &K, const Eigen::Matrix4d &Rt, int width, int height) {
            /// Set up K, Rt matrices and their inverses
            this->Rt_ = Rt;
            this->Rt_inv_ = Rt.inverse();
            this->K_ = K;
            this->K_inv_ = K_.inverse();

            /// Compute projection matrix
            this->ApplyPoseTransform(Rt);

            /// width, height
            width_ = width;
            height_ = height;
        }

        Eigen::Vector4d Camera::CameraToWorld(const Eigen::Vector4d &p_camera) const {
            return Rt_*p_camera;
        }

        Eigen::Vector4d Camera::WorldToCamera(const Eigen::Vector4d &p_world) const {
            return Rt_inv_*p_world;
        }

        Eigen::Vector3i Camera::CameraToImage(const Eigen::Vector4d &p_camera) const {
            Eigen::Vector3d proj_point = K_*p_camera.head<3>();
            proj_point /= proj_point[2]; // Normalize - last value must be 1.
            return proj_point.cast<int>();
        }

        Eigen::Vector3i Camera::WorldToImage(const Eigen::Vector4d &p_world) const {
            Eigen::Vector4d p_camera = this->WorldToCamera(p_world);
            Eigen::Vector3d p_camera_3 = p_camera.head(3);
            Eigen::Vector3d proj_point = K_*p_camera_3;

            proj_point /= proj_point[2]; // Normalize - last value must be 1.
            return proj_point.cast<int>();
        }

        Eigen::Vector4d Camera::GetCameraPose() const {
            return Rt_*reference_pose_;
        }

        Ray Camera::GetRayCameraSpace(int u, int v) const {
            Ray ray;
            ray.origin = Eigen::Vector3d(0.0,0.0,0.0);
            ray.direction = (K_inv_*Eigen::Vector3d(u,v,1.0)).normalized();
            return ray;
        }

        bool Camera::IsPointInFrontOfCamera(const Eigen::Vector4d& point) const {
            Eigen::Vector4d camera_pose = GetCameraPose();
            Eigen::Vector3d camera_plane_normal = R().col(2); //row(2);
            double camera_plane_d = 0;
            camera_plane_d -= camera_pose[0]*camera_plane_normal[0];
            camera_plane_d -= camera_pose[1]*camera_plane_normal[1];
            camera_plane_d -= camera_pose[2]*camera_plane_normal[2];

            if((camera_plane_normal.dot(point.head<3>()) + camera_plane_d) > 0)
                return true;
            else
                return false;
        }

        // Setters / Getters
        const Eigen::Matrix3d& Camera::K() const {
            return K_;
        }

        const Eigen::Matrix3d& Camera::K_inv() const {
            return K_inv_;
        }

        Eigen::Matrix3d Camera::R() const {
            return Rt_.block<3,3>(0,0);
        }

        const Eigen::Matrix4d& Camera::Rt() const {
            return Rt_;
        }

        const Eigen::Matrix4d& Camera::Rt_inv() const {
            return Rt_inv_;
        }

        const Eigen::Matrix4d& Camera::P() const {
            return P_;
        }

        int Camera::width() const {
            return width_;
        }

        int Camera::height() const {
            return height_;
        }

        void Camera::set_ground_model(const std::shared_ptr<GroundModel> ground_model) {
            ground_model_ = ground_model;
        }

        const std::shared_ptr<GroundModel> Camera::ground_model() const {
            return ground_model_;
        }

        Eigen::Vector4d Camera::PlaneCameraToWorld(const Eigen::Vector4d &plane_in_camera_space) const {
            Eigen::Vector3d t_cam = this->Rt().col(3).head<3>();
            Eigen::Vector3d plane_n_world = this->R()*plane_in_camera_space.head<3>();
            double plane_d_world = plane_in_camera_space[3] - plane_n_world.dot(t_cam);
            plane_n_world.normalize();

            Eigen::Vector4d plane_in_world_space;
            plane_in_world_space.head<3>() = plane_n_world;
            plane_in_world_space[3] = plane_d_world;
            return plane_in_world_space;
        }

        double Camera::ComputeGroundPlaneHeightInWorldSpace(double pose_plane_x, double pose_plane_z, const Eigen::Vector4d &plane_camera_space) const {
            Eigen::Vector4d plane_world = this->PlaneCameraToWorld(plane_camera_space);
            const double plane_d_world = plane_world[3];
            double a = plane_world[0];
            double b = plane_world[1];
            double c = plane_world[2];

            // Simply solve for height.
            double height_world = (-plane_d_world - a * pose_plane_x - c * pose_plane_z) / b;
            return height_world;
        }

        Eigen::Vector3d Camera::ProjectPointToGroundInWorldSpace(const Eigen::Vector3d &p_world, const Eigen::Vector4d &plane_camera_space) const {
            Eigen::Vector4d plane_world = this->PlaneCameraToWorld(plane_camera_space);
            const double plane_d_world = plane_world[3];
            double a = plane_world[0];
            double b = plane_world[1];
            double c = plane_world[2];
            double d = plane_world[3];

            Eigen::Vector3d p_proj = p_world;
            double dist = DistancePointToGroundInWorldSpace(p_world, plane_camera_space); // This is correct (bud bad design)
            p_proj -= dist*Eigen::Vector3d(a,b,c); // Substract plane_normal*dist_to_plane, and we have a projected point!

            return p_proj;
        }

        double Camera::DistancePointToGroundInWorldSpace(const Eigen::Vector3d &p_world, const Eigen::Vector4d &plane_camera_space) const {
            Eigen::Vector4d plane_world = this->PlaneCameraToWorld(plane_camera_space);
            const double plane_d_world = plane_world[3];
            double a = plane_world[0];
            double b = plane_world[1];
            double c = plane_world[2];
            double d = plane_world[3];
            Eigen::Vector3d p_proj = p_world;

            // --------------
            double x = p_world[0];
            double y = p_world[1];
            double z = p_world[2];
            double dist = a * x + b * y + c * z + d; //(std::abs(a * x + b * y + c * z + d)) /
            // --------------

            return dist;
        }

        const double Camera::f_u() const {
            return K_(0,0);
        }

        const double Camera::f_v() const {
            return K_(1,1);
        }

        const double Camera::c_u() const {
            return K_(0,2);
        }

        const double Camera::c_v() const {
            return K_(1,2);
        }

        void Camera::ApplyPoseTransform(const Eigen::Matrix4d &Rt) {
            this->Rt_ = Rt;
            this->Rt_inv_ = Rt.inverse();
        }
    }
}

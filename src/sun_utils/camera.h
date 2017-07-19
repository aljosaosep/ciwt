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

#ifndef GOT_CAMERA_H
#define GOT_CAMERA_H

// std
#include <memory>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// Forward declarations.
namespace SUN { namespace utils { class GroundModel; } }

namespace SUN {
    namespace utils {

        struct Ray {
            Eigen::Vector3d origin;
            Eigen::Vector3d direction;
        };

        /**
           * @brief Camera class. Stores camera intrinsics, extrinsics and provides some conveniance functions.
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class Camera {
        public:
            Camera();

            /**
               * @brief This method initializes camera: K, [R|t] and their inverse matrices, reference pose, width, height.
               *        With this initialization method, reference_projection_matrix is assumed to be proj. matrix in the reference frame,
               *        which is K*I 3x4 matrix. odometry_matrix represents camera-frame transformation from reference frame to current frame.
               *        This matrix can be obtained e.g. using visual odometry method (or any other approach that provides robot's odometry).
               */
            void init(const Eigen::Matrix<double,3,4> &reference_projection_matrix, const Eigen::Matrix4d &odometry_matrix, int width, int height);
            void init(const Eigen::Matrix3d &K, const Eigen::Matrix4d &odometry_matrix, int width, int height);

            /**
           * @brief Updates camera's internal state (pose)
           */
            void ApplyPoseTransform(const Eigen::Matrix4d &Rt);

            /**
               * @brief Maps point from camera coord. frame to global (reference) frame.
               * @return Transformed point.
               * @todo Do we need to take scene_Rectification_transform_ into account?
               */
            Eigen::Vector4d CameraToWorld(const Eigen::Vector4d &p_camera) const;

            /**
                * @brief Projects a plane, specified in local camera coord. sys. to 'world' or reference coord. sys.
                * @return Plane params (A, B, C, D; normal and dist.) in world space.
                */
            Eigen::Vector4d PlaneCameraToWorld(const Eigen::Vector4d &plane_in_camera_space) const;

            /**
              * @brief Very useful function, given 2D pose (world space) on specified ground plane (given in camera space),
              * it returns height of this point in world coord. frame.
              * Note: pose_plane_x, pose_plane_z are given in world coord. frame, not camera!
              *
              * @return Returns y-component (height) of this point in world coord. space.
              */
            double ComputeGroundPlaneHeightInWorldSpace(double pose_plane_x, double pose_plane_z, const Eigen::Vector4d &plane_camera_space) const;

            Eigen::Vector3d ProjectPointToGroundInWorldSpace(const Eigen::Vector3d &p_world, const Eigen::Vector4d &plane_camera_space) const;

            double DistancePointToGroundInWorldSpace(const Eigen::Vector3d &p_world, const Eigen::Vector4d &plane_camera_space) const;

            /**
               * @brief Maps point from world (reference) coord. frame to this camera frame.
               * @return Transformed point.
               * @todo Do we need to take scene_Rectification_transform_ into account?
               */
            Eigen::Vector4d WorldToCamera(const Eigen::Vector4d &p_world) const;

            /**
               * @brief Project a 3d point, given in camera space, to 2d image plane and return image coordinates.
               *        In principle, this would we only multiplication with K matrix, however, our scene cloud might be transformed/rectified,
               *        so we take scene_rectification_transformation into account as well.
               * @return Transformed point.
               */
            Eigen::Vector3i CameraToImage(const Eigen::Vector4d &p_camera) const;

            Eigen::Vector3i WorldToImage(const Eigen::Vector4d &p_world) const;

            /**
              * @brief Returns camera pose (center of proj.) in global (reference) frame.
              * @return Camera pose in world coordinates.
              */
            Eigen::Vector4d GetCameraPose() const;

            /**
               * @brief Casts ray from camera reference pose through specified pixel.
               * @return Ray structure, with origin and direction vectors.
               */
            Ray GetRayCameraSpace(int u, int v) const;

            /**
               * @brief Checks if a 3d point (given in world space) is in front of this camera plane.
               * @return True, if it is in front of camera, else otherwise.
               */
            bool IsPointInFrontOfCamera(const Eigen::Vector4d& point) const;


            /**
               * @brief TODO.
               * @return TODO.
               */
            static bool ComputeMeasurementCovariance3d(const Eigen::Vector3d &pos_3d, Eigen::Matrix2d c2d_mat, const Eigen::Matrix<double, 3, 4>& P_left_cam, const Eigen::Matrix<double, 3, 4>& P_right_cam,  Eigen::Matrix3d &covariance_out);

            /**
               * @brief TODO.
               * @return TODO.
               */
            static bool ComputeMeasurementCovariance3d(const Eigen::Vector3d &pos_3d, double c_2d, const Eigen::Matrix<double, 3, 4>& P_left_cam, const Eigen::Matrix<double, 3, 4>& P_right_cam,  Eigen::Matrix3d &covariance_out);

            // Getters / Setters
            const Eigen::Matrix3d& K() const;
            const Eigen::Matrix3d& K_inv() const;
            Eigen::Matrix3d R() const;
            int width() const;
            int height() const;

            const Eigen::Matrix4d& Rt() const;
            const Eigen::Matrix4d& Rt_inv() const;
            const Eigen::Matrix4d& P() const;

            const std::shared_ptr<GroundModel> ground_model() const;
            void set_ground_model(const std::shared_ptr<GroundModel> ground_model);

            const double f_u() const;
            const double f_v() const;
            const double c_u() const;
            const double c_v() const;

        private:
            Eigen::Matrix4d Rt_; // Extrinsic [R|t] matrix
            Eigen::Matrix4d P_; // Projection matrix: P = K*[R|t]

            Eigen::Matrix4d Rt_inv_; // Inverse pf [R|t]

            Eigen::Matrix3d K_; // Intrinsic matrix K
            Eigen::Matrix3d K_inv_; // Inverse of intrinsic matrix

            Eigen::Vector4d reference_pose_; // Reference pose: initial pose of the camera (in the first frame)

            std::shared_ptr<GroundModel> ground_model_; // Represents scene 'ground' wrt. camera.

            int width_;
            int height_;
        };
    }
}

#endif

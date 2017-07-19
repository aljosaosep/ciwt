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

// tracking
#include <tracking/visualization.h>
#include <tracking/utils_tracking.h>

// std
#include <algorithm>
#include <iostream>
#include <memory>
#include <functional>

// OpenCV
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>

// pcl
#include <pcl/common/transforms.h>

// Utils
#include "sun_utils/utils_common.h"
#include "sun_utils/utils_io.h"
#include "sun_utils/utils_visualization.h"
#include "sun_utils/ground_model.h"
#include "sun_utils/detection.h"

#define MAX_PATH_LEN 500

namespace GOT {
    namespace tracking {

        Visualizer::Visualizer() {

        }

        auto EigenMatToCvMat = [](const Eigen::Matrix2d &cov_mat_pose) -> cv::Mat {
            cv::Mat cv_cov_mat(cv::Size(2, 2), CV_64F); // row, col
            cv_cov_mat.at<double>(0, 0) = cov_mat_pose(0, 0);
            cv_cov_mat.at<double>(0, 1) = cov_mat_pose(0, 1);
            cv_cov_mat.at<double>(1, 0) = cov_mat_pose(1, 0);
            cv_cov_mat.at<double>(1, 1) = cov_mat_pose(1, 1);
            return cv_cov_mat;
        };

        std::vector<Eigen::Vector3d> Visualizer::ComputeOrientedBoundingBoxVertices(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera) {

            if (hypo.bounding_boxes_3d().size()<1)
                return std::vector<Eigen::Vector3d>();

            // Get last bounding-box
            auto bb3d = hypo.bounding_boxes_3d().back();
            Eigen::Vector4d center;
            center.head(3) = bb3d.head(3);
            center[3] = 1.0;


            // Orientation: either use velocity dir, or axis aligned (if object assumed static).
            // Not great, but looks ok-ish
            Eigen::Vector2d velocity_dir = hypo.kalman_filter_const()->GetVelocityGroundPlane();
            Eigen::Vector3d dir(/*-*/velocity_dir[0], 0.0, velocity_dir[1]);
            dir = camera.R().transpose()*dir;

            if (hypo.kalman_filter_const()->GetVelocityGroundPlane().norm() < 2.0) {
                dir = Eigen::Vector3d(0.0, 0.0, -1.0); // Is there better way to get the orientation that just set it fixed?
            }

            // Dimensions
            auto width = bb3d[3];
            auto height = bb3d[4];
            auto length = bb3d[5];

            /// Bounding box is defined by up-vector, direction vector and a vector, orthogonal to the two.
            Eigen::Vector3d ground_plane_normal = camera.ground_model()->Normal(center.head<3>()); //(0.0,-1.0,0.0);
            Eigen::Vector3d ort_to_dir_vector = dir.cross(ground_plane_normal);
            Eigen::Vector3d center_proj_to_ground_plane = center.head(3);
            center_proj_to_ground_plane = camera.ground_model()->ProjectPointToGround(center_proj_to_ground_plane);

            // Re-compute dir and or vectors
            Eigen::Vector3d dir_recomputed = ort_to_dir_vector.cross(ground_plane_normal); //ort_to_dir_vector.cross(ground_plane_normal);
            Eigen::Vector3d ort_recomputed = ground_plane_normal.cross(dir_recomputed);

            // Scale these vectors by bounding-box dimensions
            Eigen::Vector3d dir_scaled = dir_recomputed.normalized() * (length/2.0) * -1.0;
            Eigen::Vector3d ort_scaled = ort_recomputed.normalized() * (width/2.0);
            Eigen::Vector3d ground_plane_normal_scaled = ground_plane_normal*height;

            std::vector<Eigen::Vector3d> rect_3d_points(8);
            std::vector<int> rect_3d_visibility_of_points(8);

            /// Render 3d bounding-rectangle into the image
            // Compute 8 corner of a rectangle
            rect_3d_points.at(0) = center_proj_to_ground_plane + dir_scaled + ort_scaled;
            rect_3d_points.at(1) = center_proj_to_ground_plane - dir_scaled + ort_scaled;
            rect_3d_points.at(2) = center_proj_to_ground_plane + dir_scaled - ort_scaled;
            rect_3d_points.at(3) = center_proj_to_ground_plane - dir_scaled - ort_scaled;

            rect_3d_points.at(4) = center_proj_to_ground_plane + dir_scaled + ort_scaled + ground_plane_normal_scaled;
            rect_3d_points.at(5) = center_proj_to_ground_plane - dir_scaled + ort_scaled + ground_plane_normal_scaled;
            rect_3d_points.at(6) = center_proj_to_ground_plane + dir_scaled - ort_scaled + ground_plane_normal_scaled;
            rect_3d_points.at(7) = center_proj_to_ground_plane - dir_scaled - ort_scaled + ground_plane_normal_scaled;

            return rect_3d_points;
        }

        const void Visualizer::GetColor(int index, double &r, double &g, double &b) const {
            uint8_t ri, bi, gi;
            SUN::utils::visualization::GenerateColor(index, bi, gi, ri);

            b = static_cast<double>(ri)/255.0;
            g = static_cast<double>(gi)/255.0;
            r = static_cast<double>(bi)/255.0;
        }

        void Visualizer::DrawObservations(const std::vector<GOT::tracking::Observation> &observations, cv::Mat &ref_img,
                                          const SUN::utils::Camera &cam, DrawObsFnc draw_obs_fnc) const {
            auto observations_copy = observations;
            std::sort(observations_copy.begin(), observations_copy.end(),
                      [](const GOT::tracking::Observation &o1, const GOT::tracking::Observation &o2){ return o1.score()<o2.score(); });


            for (int i=0; i<observations.size(); i++) {
                draw_obs_fnc(observations.at(i), cam, ref_img, i);
            }
        }


        void Visualizer::DrawTrajectoriesBirdEye(const GOT::tracking::HypothesesVector &hypos,
                                                 pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                                 const SUN::utils::Camera &camera, cv::Mat &ref_image) const {

            // For scaling / flipping cov. matrices
            Eigen::Matrix2d flip_mat;
            flip_mat << visualization_propeties_.birdeye_scale_factor_ * 1.0, 0, 0, visualization_propeties_.birdeye_scale_factor_ * (/*-*/1.0);
            Eigen::Matrix2d world_to_cam_mat;
            const Eigen::Matrix4d &ref_to_rt_inv = camera.Rt_inv();
            world_to_cam_mat << ref_to_rt_inv(0, 0), ref_to_rt_inv(2, 0), ref_to_rt_inv(0, 2), ref_to_rt_inv(2, 2);
            flip_mat = flip_mat * world_to_cam_mat;

            // Params
            const int line_width = 2;

            ref_image = cv::Mat(visualization_propeties_.birdeye_scale_factor_*visualization_propeties_.birdeye_far_plane_,
                                (-visualization_propeties_.birdeye_left_plane_+visualization_propeties_.birdeye_right_plane_)*visualization_propeties_.birdeye_scale_factor_, CV_32FC3);
            ref_image.setTo(cv::Scalar(1.0, 1.0, 1.0));

            /// Project 3d point-cloud points to the ground and draw them.
            this->DrawPointCloudProjectionGroundPlane(point_cloud, camera, ref_image);

            /// Draw grid
            //this->DrawGridBirdeye(1.0, 1.0, camera, ref_image);

            /// Draw trajectories
            for (const auto &hypo:hypos) {

                uint8_t r, g, b;
                SUN::utils::visualization::GenerateColor(hypo.id(), r, g, b);
                auto cv_color_triplet = cv::Scalar(static_cast<float>(r)/255.0,static_cast<float>(g)/255.0, static_cast<float>(b)/255.0);

                // Draw trajectory
                cv::Point hypo_first_pose;
                cv::Point hypo_second_pose;

                const int num_poses_to_draw = 20;
                std::vector<Eigen::Vector4d> poses_copy = GOT::tracking::utils::SmoothTrajectoryPoses(hypo.poses(), 10);
                const int start_pose = std::max(1, static_cast<int>(poses_copy.size()-num_poses_to_draw));

                for (int i=start_pose; i<poses_copy.size(); i++) {
                    // Take two consecutive poses, project them to camera space
                    auto pose1 = poses_copy.at(i-1);
                    auto pose2 = poses_copy.at(i);
                    if (camera.IsPointInFrontOfCamera(pose1) && camera.IsPointInFrontOfCamera(pose2)) {
                        pose1 = camera.WorldToCamera(pose1);
                        pose2 = camera.WorldToCamera(pose2);

                        hypo_first_pose = cv::Point((pose1[0]-visualization_propeties_.birdeye_left_plane_)*visualization_propeties_.birdeye_scale_factor_, pose1[2]*visualization_propeties_.birdeye_scale_factor_);
                        hypo_second_pose = cv::Point((pose2[0]-visualization_propeties_.birdeye_left_plane_)*visualization_propeties_.birdeye_scale_factor_, pose2[2]*visualization_propeties_.birdeye_scale_factor_);

                        if (cv::clipLine(cv::Size(ref_image.cols, ref_image.rows), hypo_first_pose, hypo_second_pose))
                            cv::line(ref_image, hypo_first_pose, hypo_second_pose, cv_color_triplet, line_width);
                    }
                }

                /// Draw a circle at the last hypo pose, velocity vec. and pose covariance
                if (hypo_second_pose.x>0 && hypo_second_pose.y>0 && hypo_second_pose.x<ref_image.cols && hypo_second_pose.y<ref_image.rows) {
                    //cv::circle(ref_image, hypo_second_pose, 2.0, cv::Scalar(1.0, 0.0, 0.0), -1.0);

                    //! Draw the velocity vector
                    const Eigen::Vector2d &velocity = hypo.kalman_filter_const()->GetVelocityGroundPlane() * 0.1; // !!!
                    Eigen::Vector3d dir(velocity[0], 0.0, velocity[1]);
                    dir = camera.R().transpose()*dir; // World -> Cam

                    double x_1 = hypo.poses_camera_space().back()[0];
                    double z_1 = hypo.poses_camera_space().back()[2];

                    double x_2 = x_1 + dir[0];
                    double z_2 = z_1 + dir[2];

                    this->TransformPointToScaledFrustum(x_1, z_1); //velocity[0], velocity[1]);
                    this->TransformPointToScaledFrustum(x_2, z_2); //velocity[0], velocity[1]);

                    cv::circle(ref_image, cv::Point(x_1, z_1),  1.0, cv::Scalar(1.0, 0.0, 0.0), -1.0);
                    SUN::utils::visualization::ArrowedLine(cv::Point(x_1, z_1), cv::Point(x_2, z_2), cv::Scalar(1.0, 0.0, 0.0), ref_image);

                    // Pose cov.
                    Eigen::Matrix2d cov_mat =  hypo.kalman_filter_const()->GetPoseCovariance();
                    cov_mat = flip_mat.transpose() * cov_mat * flip_mat;
                    cv::Mat cv_cov = EigenMatToCvMat(cov_mat);
                    //SUN::utils::visualization::DrawCovarianceMatrix2d(5.991, cv::Point(x_1, z_1), cv_cov, ref_image, cv::Vec3f(1.0, 0.0, 0.0));
                }

                /// Draw a rectangle
                std::vector<Eigen::Vector3d> bbox_3d = ComputeOrientedBoundingBoxVertices(hypo, camera);
                std::vector<cv::Point2f> bbox_gp;

                for (int i=0; i<4; i++) {
                    double x = bbox_3d[i][0];
                    double z = bbox_3d[i][2];
                    TransformPointToScaledFrustum(x, z);
                    bbox_gp.push_back(cv::Point2f(x, z));
                }

                // First four vertices define the rect
                int bbox_thick = 2;
                cv::line(ref_image, bbox_gp.at(0), bbox_gp.at(1), cv_color_triplet, bbox_thick);
                cv::line(ref_image, bbox_gp.at(1), bbox_gp.at(3), cv_color_triplet, bbox_thick);
                cv::line(ref_image, bbox_gp.at(3), bbox_gp.at(2), cv_color_triplet, bbox_thick);
                cv::line(ref_image, bbox_gp.at(2), bbox_gp.at(0), cv_color_triplet, bbox_thick);
            }

            // Flip image, because it is more intuative to have ref. point at the bottom of the image
            cv::Mat dst;
            cv::flip(ref_image, dst, 0);
            ref_image = dst;
        }


        void Visualizer::DrawGridBirdeye(double res_x, double res_z, const SUN::utils::Camera &camera, cv::Mat &ref_image) const {


            auto color = cv::Scalar(0.5, 0.5, 0.5);

            // Draw horizontal lines
            for (double i=0; i<visualization_propeties_.birdeye_far_plane_; i+=res_z) {
                double x_1 = visualization_propeties_.birdeye_left_plane_;
                double y_1 = i;

                double x_2 = visualization_propeties_.birdeye_right_plane_;
                double y_2 = i;

                this->TransformPointToScaledFrustum(x_1, y_1);
                this->TransformPointToScaledFrustum(x_2, y_2);

                auto p1 = cv::Point(x_1, y_1), p2=cv::Point(x_2,y_2);
                cv::line(ref_image, p1, p2, color);
            }

            // Draw vertical lines
            for (double i=visualization_propeties_.birdeye_left_plane_; i<visualization_propeties_.birdeye_right_plane_; i+=res_x) {
                double x_1 = i;
                double y_1 = 0;

                double x_2 = i;
                double y_2 = visualization_propeties_.birdeye_far_plane_;

                this->TransformPointToScaledFrustum(x_1, y_1);
                this->TransformPointToScaledFrustum(x_2, y_2);

                auto p1 = cv::Point(x_1, y_1), p2=cv::Point(x_2,y_2);
                cv::line(ref_image, p1, p2, color);
            }

        }

        void Visualizer::DrawPointCloudProjectionGroundPlane(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                                             const SUN::utils::Camera &camera, cv::Mat &ref_image) const {
            // Project 3d point-cloud points to the ground and draw them.
            for (const auto &p:point_cloud->points) {
                if (!std::isnan(p.x)) {
                    Eigen::Vector3d p_proj_to_gp = camera.ground_model()->ProjectPointToGround(p.getVector3fMap().cast<double>());
                    double pose_x = p_proj_to_gp[0];
                    double pose_z = p_proj_to_gp[2];
                    if (pose_x > visualization_propeties_.birdeye_left_plane_ && pose_x < visualization_propeties_.birdeye_right_plane_ && pose_z > 0.0 && pose_z < visualization_propeties_.birdeye_far_plane_) {
                        TransformPointToScaledFrustum(pose_x,pose_z);
                        ref_image.at<cv::Vec3f>(static_cast<int>(pose_z), static_cast<int>(pose_x)) -= cv::Vec3f(0.1, 0.1, 0.1);
                    }
                }
            }
        }

        void Visualizer::TransformPointToScaledFrustum(double &pose_x, double &pose_z) const {
            pose_x += (-visualization_propeties_.birdeye_left_plane_);
            pose_x *= visualization_propeties_.birdeye_scale_factor_;
            pose_z *= visualization_propeties_.birdeye_scale_factor_;
        }

        void Visualizer::DrawObservationsBirdEye(const std::vector<GOT::tracking::Observation> &observations,
                                                 pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                                 const SUN::utils::Camera &camera, cv::Mat &ref_image) const {

            ref_image = cv::Mat(visualization_propeties_.birdeye_scale_factor_*visualization_propeties_.birdeye_far_plane_,
                                (-visualization_propeties_.birdeye_left_plane_+visualization_propeties_.birdeye_right_plane_)*visualization_propeties_.birdeye_scale_factor_, CV_32FC3);
            ref_image.setTo(cv::Scalar(1.0, 1.0, 1.0));

            // Project 3d point-cloud points to the ground and draw them.
            DrawPointCloudProjectionGroundPlane(point_cloud, camera, ref_image);

            this->DrawGridBirdeye(1.0, 1.0, camera, ref_image);

            // Draw observations
            int obs_index = 0;
            for (const auto &obs:observations) {
                const auto &pose3d = obs.footpoint();
                auto obs_cov_3d = obs.covariance3d();

                // I apply flip trans. on the covariance, because of the image and scene-scaling I do latter on
                Eigen::Matrix3d flip_mat;
                flip_mat << visualization_propeties_.birdeye_scale_factor_*1.0, 0, 0, 0, visualization_propeties_.birdeye_scale_factor_*1.0, 0, 0, 0, visualization_propeties_.birdeye_scale_factor_*(/*-*/1.0);
                obs_cov_3d = flip_mat.transpose()*obs_cov_3d*flip_mat;

                const auto &footp = camera.ground_model()->ProjectPointToGround(pose3d.head<3>());
                double pose_x = footp[0];
                double pose_z = footp[2];

                if (pose_x > visualization_propeties_.birdeye_left_plane_ && pose_x < visualization_propeties_.birdeye_right_plane_ && pose_z > 0.0 && pose_z < visualization_propeties_.birdeye_far_plane_) {
                    TransformPointToScaledFrustum(pose_x, pose_z);
                    auto pose_xz_cv = cv::Point(pose_x, pose_z);

                    // Visualize 2d covariance as oriented box
                    cv::Mat cov_2d(cv::Size(2, 2), CV_64F); // row, col
                    cov_2d.at<double>(0,0) = obs_cov_3d(0,0);
                    cov_2d.at<double>(0,1) = obs_cov_3d(0,2);
                    cov_2d.at<double>(1,0) = obs_cov_3d(2,0);
                    cov_2d.at<double>(1,1) = obs_cov_3d(2,2);

                    double r,g,b;
                    this->GetColor(obs_index++, r,g,b);

                    const cv::Vec3f color(r,g,b);
                    SUN::utils::visualization::DrawCovarianceMatrix2dEllipse/*Smooth*/(5.991, cv::Point2f(static_cast<int>(pose_x), static_cast<int>(pose_z)),
                                                                                       cov_2d, ref_image, color);
                }
            }

            // Draw observations
            obs_index = 0;
            for (const auto &obs:observations) {

                const auto &pose3d = obs.footpoint();

                // Cue-avalability determines the color of the dot!
                cv::Scalar color_of_the_dot(0,0,1.0);

                const auto &footp = camera.ground_model()->ProjectPointToGround(pose3d.head<3>());
                double pose_x = footp[0];
                double pose_z = footp[2];

                if (pose_x > visualization_propeties_.birdeye_left_plane_ && pose_x < visualization_propeties_.birdeye_right_plane_ && pose_z > 0.0 && pose_z < visualization_propeties_.birdeye_far_plane_) {
                    TransformPointToScaledFrustum(pose_x, pose_z);

                    auto pose_xz_cv = cv::Point(pose_x, pose_z);

                    // Visualize 3d-velocity measurement.
                    auto velocity_arrow = obs.velocity()/**0.1*/;
                    SUN::utils::visualization::ArrowedLine(pose_xz_cv,
                                                           cv::Point2d(pose_xz_cv.x + velocity_arrow[0],
                                                                       pose_xz_cv.y + velocity_arrow[2]),
                                                           cv::Scalar(0, 0, 255), ref_image,
                                                           2, 8, 0, 0.2);

                    // Visualize pose measurement.
                    cv::circle(ref_image, pose_xz_cv, 3.0, /*cv::Scalar(0,0,255)*/color_of_the_dot, -1);

                    auto transform_point_to_scaled_sys = [this](
                            const Eigen::Vector2d &p) -> cv::Point2d {
                        auto x = p[0];
                        auto y = p[1];

                        this->TransformPointToScaledFrustum(x, y);

                        return cv::Point2d(x, y);
                    };

                    // Draw obs. id
                    cv::putText(ref_image, std::to_string(obs_index),
                                cv::Point(pose_x + 5, pose_z -5),
                                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 255), 1);
                }

                obs_index ++;
            }

            // Flip image, because it is more intuitive to have ref. point at the bottom of the image
            cv::Mat dst;
            cv::flip(ref_image, dst, 0);
            ref_image = dst;
        }

        void Visualizer::DrawHypotheses(const std::vector<GOT::tracking::Hypothesis> &hypos, const SUN::utils::Camera &camera, cv::Mat &ref_image, DrawHypoFnc draw_hypo_fnc) const {
            GOT::tracking::HypothesesVector hypos_copy = hypos;
            std::sort(hypos_copy.begin(), hypos_copy.end(), [](const GOT::tracking::Hypothesis& i, const GOT::tracking::Hypothesis& j)
            { return (i.poses_camera_space().back()[2])>(j.poses_camera_space().back()[2]); });

            for (const auto&hypo:hypos_copy) {
                draw_hypo_fnc(hypo, camera, ref_image);
            }
        }

        void Visualizer::RenderHypo3D(pcl::visualization::PCLVisualizer &viewer, const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,  const int viewport) {

            if (hypo.poses().size()<2)
                return;

            // Params
            const int line_width = 7;
            const int num_poses_to_draw = 200;

            // Get R,G,B triplet from color lookup table.
            double r,g,b;
            GetColor(hypo.id(), b, g, r);

            // Center
            if (hypo.bounding_boxes_3d().size()<1)
                return;

            const Eigen::VectorXd bb3d = hypo.bounding_boxes_3d().back();
            Eigen::Vector4d center;
            center << bb3d.head(3), 1.0;

            std::vector<Eigen::Vector3d> rect_3d_points = ComputeOrientedBoundingBoxVertices(hypo, camera);

            std::vector<pcl::PointXYZRGBA> pcl_points(rect_3d_points.size());
            for (int i=0; i<8; i++)
                pcl_points.at(i).getVector3fMap() = rect_3d_points.at(i).cast<float>();

            int idx = 0;
            std::string line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(0),pcl_points.at(1), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(1),pcl_points.at(3), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(3),pcl_points.at(2), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(2),pcl_points.at(0), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(4),pcl_points.at(5), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(5),pcl_points.at(7), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(7),pcl_points.at(6), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(6),pcl_points.at(4), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(0),pcl_points.at(4), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(1),pcl_points.at(5), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(2),pcl_points.at(6), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);

            line_id = "hypo_"+std::to_string(hypo.id())+"_line_"+std::to_string(idx++);
            viewer.addLine(pcl_points.at(3),pcl_points.at(7), r, g, b, line_id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, line_id, viewport);
        }

        void Visualizer::RenderTrajectory(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera, const std::string &traj_id, double r, double g, double b, pcl::visualization::PCLVisualizer &viewer,  int viewport) {
            const std::vector<Eigen::Vector4d> &hypo_traj = hypo.poses();

            std::vector<Eigen::Vector4d> poses_copy = GOT::tracking::utils::SmoothTrajectoryPoses(hypo_traj, 20);
            const int start_pose = std::max(1, static_cast<int>(poses_copy.size()-40));
            for (int j=start_pose; j<poses_copy.size(); j++) {
                //for (unsigned j = 1; j < hypo_traj.size(); j++) {
                std::string pt_id = traj_id + std::to_string(j) + "_" +std::to_string(j);
                pcl::PointXYZRGBA p1, p2;

                Eigen::Vector4d p1_eig = poses_copy[j - 1];
                Eigen::Vector4d p2_eig = poses_copy[j];

                p1_eig[1] -= 0.1;
                p2_eig[1] -= 0.1;

                p1_eig = camera.WorldToCamera(p1_eig);
                p2_eig = camera.WorldToCamera(p2_eig);
                p1.getVector3fMap() = p1_eig.head<3>().cast<float>();
                p2.getVector3fMap() = p2_eig.head<3>().cast<float>();
                viewer.addLine(p1, p2, r, g, b, pt_id, viewport);
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4.0, pt_id, viewport);
            }

        }

        void Visualizer::set_visualization_propeties(const Visualizer::VisualizationProperties &visualization_props) {
            this->visualization_propeties_ = visualization_props;
        }

        namespace draw_hypos {

            void DrawTrajectoryToGroundPlane(const std::vector<Eigen::Vector4d> &poses,  const SUN::utils::Camera &camera, const cv::Scalar &color, cv::Mat &ref_image, int line_width, int num_poses_to_draw, int smoothing_window_size) {
                std::vector<Eigen::Vector4d> poses_copy = GOT::tracking::utils::SmoothTrajectoryPoses(poses, smoothing_window_size);
                const int start_pose = std::max(1, static_cast<int>(poses_copy.size()-num_poses_to_draw));
                for (int i=start_pose; i<poses_copy.size(); i++) {
                    // Take two consecutive poses, project them to camera space
                    auto pose1 = poses_copy.at(i-1);
                    auto pose2 = poses_copy.at(i);
                    if (camera.IsPointInFrontOfCamera(pose1) && camera.IsPointInFrontOfCamera(pose2) /*&& at_least_one_stereo_meas_avalible*/ /*&& (stereo_avalible_p1 || stereo_avalible_p2)*/) {
                        pose1 = camera.WorldToCamera(pose1);
                        pose2 = camera.WorldToCamera(pose2);
                        SUN::utils::visualization::DrawLine(pose1.head<3>(), pose2.head<3>(), camera, ref_image, color, line_width, 1);
                    }
                }
            }

            void DrawHypothesis2d(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera, cv::Mat &ref_image) {

                if (hypo.terminated())
                    return;

                // Params
                const int line_width = 3;
                const int num_poses_to_draw = 20;

                // Get R,G,B triplet from color lookup table.
                int col_id = hypo.id();
                if (hypo.id()<0)
                    col_id=0;

                uint8_t r,g,b;
                SUN::utils::visualization::GenerateColor(col_id, r, g, b); // RGB, BGR!
                auto cv_color_triplet = cv::Scalar(r, g, b);

                /// Draw trajectory
                DrawTrajectoryToGroundPlane(hypo.poses(), camera, cv_color_triplet, ref_image, 2.0, num_poses_to_draw, 10);

                /// Draw last bounding-box 2d.
                auto bb2d = (hypo.bounding_boxes_2d_with_timestamp().rbegin()->second);
                cv::Rect rect(cv::Point2d(bb2d[0],bb2d[1]), cv::Size(bb2d[2], bb2d[3]));
                cv::rectangle(ref_image, rect, cv_color_triplet, line_width);
                Eigen::Vector2d posterior_velocity_gp = hypo.kalman_filter_const()->GetVelocityGroundPlane();

                auto last_hypo_pose = hypo.poses().back();
                auto p1_cam_space = camera.WorldToCamera(last_hypo_pose);

                auto p1_proj = camera.CameraToImage(p1_cam_space);
                auto p1_proj_cv = cv::Point(p1_proj[0], p1_proj[1]);

                // Get velocity endpoint: p1+p2*dt. Then project it to the image.
                Eigen::Vector4d p2_cam_space;
                p2_cam_space.head<3>() = p1_cam_space.head<3>() + camera.R().transpose()*Eigen::Vector3d(posterior_velocity_gp[0], p1_cam_space[1], posterior_velocity_gp[1])*0.1;
                p2_cam_space[3] = 1.0;

                auto p2_proj = camera.CameraToImage(p2_cam_space);
                auto p2_proj_cv = cv::Point(p2_proj[0], p2_proj[1]);

                // Draw hypo id string
                cv::putText(ref_image,  std::to_string(hypo.id()), cv::Point2d(bb2d[0]+5, bb2d[1]+20), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, cv_color_triplet, 2);
//
//                // Draw category string
//                std::string category_string = "?";
//
//                if (category==SUN::shared_types::CAR)
//                    category_string = "CAR";
//                else if (category==SUN::shared_types::PEDESTRIAN)
//                    category_string = "PED.";
//                else if (category==SUN::shared_types::CYCLIST)
//                    category_string = "CYC.";
//                else if (category==SUN::shared_types::VAN)
//                    category_string = "VAN";
//                else if (category==SUN::shared_types::TRUCK)
//                    category_string = "TRUCK";
//                else if (category==SUN::shared_types::UNKNOWN_TYPE)
//                    category_string = "UNK.";
//                cv::putText(ref_image,  category_string, cv::Point2d(bb2d[0]+5, bb2d[1]+40), cv::FONT_HERSHEY_PLAIN, 0.8, cv_color_triplet, 2);
            }

            void DrawHypothesis3d(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera, cv::Mat &ref_image) {
                if (hypo.poses().size()<2)
                    return;

                // Params
                const int line_width = 3;
                const int num_poses_to_draw = 200;

                // Get R,G,B triplet from color lookup table.
                uint8_t r,g,b;
                SUN::utils::visualization::GenerateColor(hypo.id(), r, g, b); // RGB, BGR!
                auto cv_color_triplet = cv::Scalar(r, g, b);

                // Center
                if (hypo.bounding_boxes_3d().size()<1)
                    return;

                auto bb3d = hypo.bounding_boxes_3d().back();
                Eigen::Vector4d center;
                center.head(3) = bb3d.head(3);
                center[3] = 1.0;

                DrawTrajectoryToGroundPlane(hypo.poses(), camera, cv_color_triplet, ref_image, 2.0, num_poses_to_draw, 20.0);

                // Draw 3D-bbox center 3D
                Eigen::Vector3i center_in_image = camera.CameraToImage(center);
                cv::circle(ref_image, cv::Point(center_in_image[0], center_in_image[1]), 1.0, cv::Scalar(0,0,255), -1);

                std::vector<Eigen::Vector3d> rect_3d_points =  Visualizer::ComputeOrientedBoundingBoxVertices(hypo, camera);
                std::vector<int> rect_3d_visibility_of_points(8);

                // Compute point visibility
                for (int i=0; i<8; i++) {
                    Eigen::Vector4d pt_4d;
                    pt_4d.head<3>() = rect_3d_points.at(i);
                    pt_4d[3] = 1.0;
                    rect_3d_visibility_of_points.at(i) = camera.IsPointInFrontOfCamera(pt_4d);
                }

                // Render lines
                SUN::utils::visualization::DrawLine(rect_3d_points.at(0),rect_3d_points.at(1), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(1),rect_3d_points.at(3), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(3),rect_3d_points.at(2), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(2),rect_3d_points.at(0), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(4),rect_3d_points.at(5), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(5),rect_3d_points.at(7), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(7),rect_3d_points.at(6), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(6),rect_3d_points.at(4), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(0),rect_3d_points.at(4), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(1),rect_3d_points.at(5), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(2),rect_3d_points.at(6), camera, ref_image, cv_color_triplet, line_width, 1);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(3),rect_3d_points.at(7), camera, ref_image, cv_color_triplet, line_width, 1);

                double bbox_len = bb3d[5];

//                /// Draw 'dir' arrow
////                Eigen::Vector2d velocity_dir = hypo.kalman_filter_const()->GetVelocityGroundPlane();
////                Eigen::Vector3d dir(/*-*/velocity_dir[0], 0.0, velocity_dir[1]);
////                dir = camera.R().transpose()*dir;
//
//                Eigen::Vector3d velocity_dir(1.0, 0.0, 0.0);
//                Eigen::AngleAxisd yawAngle(hypo.kalman_filter_const()->x()[2], Eigen::Vector3d::UnitY());
//                Eigen::Vector3d dir=yawAngle.matrix()*velocity_dir;
//
//                // Direction arrow
//                Eigen::Vector3d arrow_end = center.head<3>()+ dir.normalized()*bbox_len;
//                SUN::utils::visualization::DrawLine(center.head<3>(), arrow_end, camera, ref_image, cv::Scalar(0,0,255), line_width, 3);
//
//                SUN::utils::visualization::DrawLine(arrow_end, arrow_end+Eigen::Vector3d(1.0, 0.0, 0.5), camera, ref_image, cv::Scalar(0,0,255), line_width, 3);
//                SUN::utils::visualization::DrawLine(arrow_end, arrow_end+Eigen::Vector3d(-1.0, 0.0, 0.5), camera, ref_image, cv::Scalar(0,0,255), line_width, 3);

                DrawTrajectoryToGroundPlane(hypo.poses(), camera, cv_color_triplet, ref_image, 2.0, 20, 10);
            }
        }

        namespace draw_observations {

            void DrawObservationAndOrientation(const GOT::tracking::Observation &observation, const SUN::utils::Camera &cam, cv::Mat &ref_img, int index) {
                DrawObservationByID(observation, cam, ref_img, index);

                if (observation.orientation()>-100.0) {

                    // Draw orientation estim. as arrowed line
                    Eigen::Vector3d orient_vec(1.0, observation.footpoint()[1], 0.0);
                    Eigen::AngleAxisd yaw_angle_tranf(observation.orientation(), Eigen::Vector3d::UnitY());

                    orient_vec = yaw_angle_tranf.toRotationMatrix() * orient_vec;
                    orient_vec *= 2.0;

                    Eigen::Vector4d orientation_vec_endpoint = observation.footpoint();
                    orientation_vec_endpoint[0] += orient_vec[0];
                    orientation_vec_endpoint[2] += orient_vec[2];

                    auto proj_pt = cam.CameraToImage(observation.footpoint());
                    auto cv_proj_pt = cv::Point(proj_pt[0], proj_pt[1]);

                    Eigen::Vector3i proj_endpoint = cam.CameraToImage(orientation_vec_endpoint);
                    SUN::utils::visualization::ArrowedLine(cv_proj_pt, cv::Point(proj_endpoint[0], proj_endpoint[1]), cv::Scalar(0, 255, 0),
                                                           ref_img, 2, 8, 0, 0.1);
                }
            }

            void DrawObservationPaper(const GOT::tracking::Observation &observation, const SUN::utils::Camera &cam, cv::Mat &ref_img, int index)  {

                auto generate_color = [](unsigned int id, uint8_t &r, uint8_t &g, uint8_t &b) {
                    if (id%10 == 0) { r = 240; g = 62; b = 36;}
                    if (id%10 == 1) { r = 245; g = 116; b = 32;}
                    if (id%10 == 2) { r = 251; g = 174; b = 24;}
                    if (id%10 == 3) { r = 213; g = 223; b = 38;}
                    if (id%10 == 4) { r = 153; g = 204; b = 112;}
                    if (id%10 == 5) { r = 136; g = 201; b = 141;}
                    if (id%10 == 6) { r = 124; g = 201; b = 169;}
                    if (id%10 == 7) { r = 100; g = 199; b = 230;}
                    if (id%10 == 8) { r = 64; g = 120; b = 188;}
                    if (id%10 == 9) { r = 61; g = 88; b = 167;}
                };

                uint8_t r, g, b;
                generate_color(index, r,g,b);

                if (observation.proposal_3d_avalible() && observation.detection_avalible()) {
                    SUN::utils::visualization::DrawObjectFilled(observation.pointcloud_indices(), observation.bounding_box_2d(),
                                                                cv::Vec3b(b,g,r), 0.5, ref_img);
                }

                else if (observation.detection_avalible() && !observation.proposal_3d_avalible()) {
                    SUN::utils::visualization::DrawObjectFilled(observation.pointcloud_indices(), observation.bounding_box_2d(),
                                                                cv::Vec3b(0, 255, 0), 0.5, ref_img);
                }
            }

            void DrawObservationByID(const GOT::tracking::Observation &observation, const SUN::utils::Camera &cam, cv::Mat &ref_img, int index)  {
                uint8_t r,g,b;
                //GetColor(index, r,g, b);
                SUN::utils::visualization::GenerateColor(index, r, g, b);
                const auto bb2d = observation.bounding_box_2d();
                SUN::utils::visualization::DrawObjectFilled(observation.pointcloud_indices(), observation.bounding_box_2d(), cv::Vec3b(r,g,b), 0.5, ref_img);
            }
        }
    }
}

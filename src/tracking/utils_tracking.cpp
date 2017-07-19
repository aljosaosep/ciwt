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

#include <tracking/utils_tracking.h>

// std
#include <iostream>

// tracking
#include <tracking/category_filter.h>

// utils
#include "sun_utils/utils_io.h"
#include "sun_utils/utils_kitti.h"
#include "sun_utils/utils_bounding_box.h"
#include "sun_utils/utils_pointcloud.h"
#include "sun_utils/ground_model.h"
#include "sun_utils/utils_observations.h"

namespace GOT {
    namespace tracking {
        namespace utils {


            Eigen::Matrix4d EstimateEgomotion(libviso2::VisualOdometryStereo &viso, const cv::Mat &color_left, const cv::Mat &color_right) {
                cv::Mat grayscale_left, grayscale_right;
                cv::cvtColor(color_left, grayscale_left, CV_BGR2GRAY);
                cv::cvtColor(color_right, grayscale_right, CV_BGR2GRAY);
                const int32_t width = color_left.cols;
                const int32_t height = color_left.rows;

                // Convert input images to uint8_t buffer
                uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
                uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
                int32_t k=0;
                for (int32_t v=0; v<height; v++) {
                    for (int32_t u=0; u<width; u++) {
                        left_img_data[k]  = (uint8_t)grayscale_left.at<uchar>(v,u);
                        right_img_data[k] = (uint8_t)grayscale_right.at<uchar>(v,u);
                        k++;
                    }
                }

                Eigen::Matrix<double,4,4> egomotion_eigen = Eigen::MatrixXd::Identity(4,4); // Current pose
                int32_t dims[] = {width,height,width};
                if (viso.process(left_img_data,right_img_data,dims)) {
                    // on success, update current pose
                    //std::vector<int32_t> inlier_indices = viso->getInlierIndices();
                   // std::vector<libviso2::Matcher::p_match> matches = viso.getMatches();
                    libviso2::Matrix frame_to_frame_motion = viso.getMotion(); //libviso2::Matrix::inv(viso->getMotion());
                    for(size_t i=0; i<4; i++){
                        for(size_t j=0; j<4; j++){
                            egomotion_eigen(i,j) = frame_to_frame_motion.val[i][j];
                        }
                    }
                }

                free(left_img_data);
                free(right_img_data);
                return egomotion_eigen;
            }


            SUN::utils::KITTI::TrackingLabel HypoToLabelDefault(int frame, const GOT::tracking::Hypothesis &hypo, double image_width,
                                                         double image_height) {

                // Compute 'alpha' angle.
                // Aljosa: don't rember at the moment why was it computed that way. It's copied from old framework.
                Eigen::Vector3d hypo_dir = hypo.GetDirectionVector();
                double alpha_angle = -std::atan2(hypo_dir[2], hypo_dir[0]) - M_PI;


                auto IndexToCategoryFnc = [](int index)->SUN::shared_types::CategoryTypeKITTI {
                    return static_cast<SUN::shared_types::CategoryTypeKITTI >(index);
                };

                SUN::shared_types::CategoryTypeKITTI categoryType = IndexToCategoryFnc(GOT::tracking::bayes_filter::GetArgMax(hypo.category_probability_distribution_));

                SUN::utils::KITTI::TrackingLabel label;
                label.frame = frame;
                label.trackId = hypo.id();
                label.type = categoryType;
                label.truncated = -1;
                label.occluded = SUN::utils::KITTI::NOT_SPECIFIED;
                label.alpha = alpha_angle;

                Eigen::Vector4d bbox_2d = hypo.bounding_boxes_2d_with_timestamp().rbegin()->second;
                label.boundingBox2D[0] = bbox_2d[0];
                label.boundingBox2D[1] = bbox_2d[1];
                label.boundingBox2D[2] = bbox_2d[2] + bbox_2d[0];
                label.boundingBox2D[3] = bbox_2d[3] + bbox_2d[1];

                // Make sure bbox2d is trunc. to the image borders
                if (label.boundingBox2D[0] < 0)
                    label.boundingBox2D[0] = 0;
                if (label.boundingBox2D[1] < 0)
                    label.boundingBox2D[1] = 0;

                if (label.boundingBox2D[2] > image_width)
                    label.boundingBox2D[2] = image_width;
                if (label.boundingBox2D[3] > image_height)
                    label.boundingBox2D[3] = image_height;

                Eigen::Vector3d pose_cam = hypo.poses_camera_space().back();
                Eigen::VectorXd bbox_3d = hypo.bounding_boxes_3d().back();

                label.dimensions[0] = bbox_3d[4];
                label.dimensions[1] = bbox_3d[3];
                label.dimensions[2] = bbox_3d[5];

                label.location[0] = pose_cam[0];
                label.location[1] = pose_cam[1];
                label.location[2] = pose_cam[2];

                label.rotationY = -10;
                label.score = hypo.score();

                assert(!std::isnan(hypo.score()));

                return label;
            }

            void HypothesisSetToLabels(int frame,
                                       const GOT::tracking::HypothesesVector &hypos,
                                       std::vector<SUN::utils::KITTI::TrackingLabel> &labels,
                                       std::function<SUN::utils::KITTI::TrackingLabel(int, const GOT::tracking::Hypothesis&)> hypo_to_label_fnc
            ) {
                for (const auto &hypo:hypos) {
                    const auto label = hypo_to_label_fnc(frame, hypo);
                    labels.push_back(label);
                }
            }


            Eigen::Vector2d ProjectEgomotion(const Eigen::Vector4d &reference_pose_camera_space,
                                             const SUN::utils::Camera &cam_current_frame,
                                             const SUN::utils::Camera &cam_previous_frame) {
                auto Rt_curr = cam_current_frame.Rt(); // Rt_curr is egomotion estimate from frame_0 to frame_current
                auto Rt_prev = cam_previous_frame.Rt();
                auto delta_Rt = Rt_prev.inverse() * Rt_curr; // Get frame-to-frame egomotion estimate
                const Eigen::Vector4d ref_point_inv = delta_Rt * reference_pose_camera_space; // Apply inverse frame-to-frame transform on ref. pt.
                Eigen::Vector3i proj_point1 = cam_current_frame.CameraToImage(reference_pose_camera_space);
                Eigen::Vector3i proj_point2 = cam_current_frame.CameraToImage(ref_point_inv);
                Eigen::Vector2d diff_vec = (proj_point1 - proj_point2).head<2>().cast<double>(); // This is our 'projected' velocity observation!
                return diff_vec;
            }

            /**
               * @brief Evaluate if a point is inside an elipse.
               * @author Author: Dennis Mitzel (mitzel@vision.rwth-aachen.de). Tested by Aljosa Nov'2015 (works correctly!) (osep@vision.rwth-aachen.de).
               */
            bool PointInEllipse(double point_x, double point_y, double window_center_x, double window_center_y, double window_w, double window_h) {
                if ((((point_x - window_center_x) * 2.0 / window_w) * ((point_x - window_center_x) * 2.0 / window_w) +
                     ((point_y - window_center_y) * 2.0 / window_h) * ((point_y - window_center_y) * 2.0 / window_h)) <= 1)
                    return true;
                return false;
            }

            void ObservationSetToLabels(int frame, const std::vector<GOT::tracking::Observation> &observations,
                                        std::vector<SUN::utils::KITTI::TrackingLabel> &labels) {
                for (const auto &obs:observations) {

                    SUN::utils::KITTI::TrackingLabel label;
                    label.frame = frame;
                    label.trackId = -1;
                    label.type = obs.detection_category();

                    label.truncated = -1;
                    label.occluded = SUN::utils::KITTI::NOT_SPECIFIED;
                    label.alpha = -1;

                    label.boundingBox2D[0] = obs.bounding_box_2d()[0];
                    label.boundingBox2D[1] = obs.bounding_box_2d()[1];
                    label.boundingBox2D[2] = obs.bounding_box_2d()[2] + obs.bounding_box_2d()[0];
                    label.boundingBox2D[3] = obs.bounding_box_2d()[3] + obs.bounding_box_2d()[1];

                    // Dim. order modified to comply with kitti (i.e. the code below is correct, even if it doesn't seem so)
                    if (obs.proposal_3d_avalible()) {
                        label.dimensions[0] = obs.bounding_box_3d()[4];
                        label.dimensions[1] = obs.bounding_box_3d()[3];
                        label.dimensions[2] = obs.bounding_box_3d()[5];
                    }
                    else {
                        label.dimensions[0] = label.dimensions[1] = label.dimensions[2] = -1;
                    }

                    label.location[0] = obs.footpoint()[0];
                    label.location[1] = obs.footpoint()[1];
                    label.location[2] = obs.footpoint()[2];

                    label.rotationY = -10;
                    label.score = obs.score();

                    labels.push_back(label);
                }
            }

            bool ComputeDetectionPoseUsingStereo(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                                 const Eigen::Vector4d &bounding_box_2d, Eigen::Vector4d &median_footpoint) {
                const int bbox_x1 = std::ceil(bounding_box_2d[0]);
                const int bbox_y1 = std::ceil(bounding_box_2d[1]);
                const int bbox_w = std::ceil(bounding_box_2d[2]);
                const int bbox_h = std::ceil(bounding_box_2d[3]);

                // Dennises magic params for ellipse fit
                const double bbox_height_cut_for_ellipse = 0.3; // Scale by that much.
                const double bbox_width_cut_for_ellipse = 0.3;
                const double new_height_for_ellipse = std::floor(bbox_h - (bbox_h * bbox_height_cut_for_ellipse));
                const double new_width_for_ellipse = std::floor(bbox_w - (bbox_w * bbox_width_cut_for_ellipse));
                const int center_x_for_ellipse = std::floor(bbox_x1 + (bbox_w / 2.0));
                const int center_y_for_ellipse = std::floor(bbox_y1 + (new_height_for_ellipse / 2.0) + new_height_for_ellipse * 0.1);

                std::vector<double> pts_det_x;
                std::vector<double> pts_det_z;
                std::vector<double> pts_det_y;

                // Consider elyptical-area
                for (int i = 0; i < bbox_h; i++) {
                    for (int j = 0; j < bbox_w; j++) {
                        int row_index = i + bbox_y1;
                        int col_index = j + bbox_x1;
                        if ((row_index > 0) && (col_index > 0) && (row_index < point_cloud->height) &&
                            (col_index < point_cloud->width)) {
                            const auto &point3d = point_cloud->at(col_index, row_index); // (col, row)
                            if (!std::isnan(point3d.x)) {
                                Eigen::Vector3d point_vec_eigen = point3d.getVector3fMap().cast<double>();
                                if (PointInEllipse(static_cast<double>(col_index), static_cast<double>(row_index),
                                                   static_cast<double>(center_x_for_ellipse),
                                                   static_cast<double>(center_y_for_ellipse),
                                                   new_width_for_ellipse, new_height_for_ellipse)) {
                                    pts_det_x.push_back(point_vec_eigen[0]);
                                    pts_det_z.push_back(point_vec_eigen[2]);
                                }

                                // For the y-coord., the point [does][should] not to be in the ellipse!
                                pts_det_y.push_back(point_vec_eigen[1]);
                            }
                        }
                    }
                }

                if (pts_det_x.size() > 5 && pts_det_z.size() > 5) {
                    std::sort(pts_det_x.begin(), pts_det_x.end());
                    std::sort(pts_det_z.begin(), pts_det_z.end());
                    std::sort(pts_det_y.begin(), pts_det_y.end());
                    double median_x = pts_det_x.at(static_cast<unsigned>(pts_det_x.size() / 2));
                    double median_z = pts_det_z.at(static_cast<unsigned>(pts_det_z.size() / 2));
                    double robust_y_coord = pts_det_y.at(std::floor(pts_det_y.size() * 0.95));
                    median_footpoint = Eigen::Vector4d(median_x, robust_y_coord, median_z, 1.0);
                    return true;
                }
                return false;
            }


            std::vector<Eigen::Vector4d> SmoothTrajectoryPoses(const std::vector<Eigen::Vector4d> &poses, int kernel_size) {
                const int width = std::floor(kernel_size/2.0);
                const int num_poses = poses.size();

                // Kernel size unreasonably small -> return original points.
                if (width <= 0)
                    return poses;

                std::vector<Eigen::Vector4d> smooth_poses;
                for (int i=0; i<num_poses; i++) {
                    const int local_window = std::min(std::min(i, width), num_poses-i-1); // Make sure we don't spill over the boundaries
                    const int left_pos = i - local_window;
                    const int right_pos = i + local_window;

                    double mean_x = 0.0;
                    for (int j=left_pos; j<=right_pos; j++)
                        mean_x += poses.at(j)[0];

                    if((right_pos-left_pos) > 0) {
                        mean_x *= (1.0/((right_pos - left_pos)+1)); // In general, divide by window_size, except if we are looking at smaller nbhd (boundaries)
                        Eigen::Vector4d smooth_pose = poses.at(i);
                        smooth_pose[0] = mean_x;
                        smooth_poses.push_back(smooth_pose);
                    }
                }

                return smooth_poses;
            }
        }
    }
}

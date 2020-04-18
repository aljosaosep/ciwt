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

#include "utils_observations.h"

// opencv
#include <opencv2/imgproc.hpp>

// utils
#include "utils_common.h"
#include "ground_model.h"


namespace SUN {
    namespace utils {
        namespace observations {

            Eigen::Vector3d ComputeVelocity(const cv::Mat &velocity_map,
                                            const std::vector<int> &indices, double dt) {

                std::vector<Eigen::Vector3d> det_flows;

                for (auto ind:indices) {
                    int x, y;
                    SUN::utils::UnravelIndex(ind, velocity_map.cols, &x, &y);
                    const cv::Vec3f &velocity_meas = velocity_map.at<cv::Vec3f>(y, x);
                    if (!std::isnan(velocity_meas[0])) {
                        Eigen::Vector3d flow_vec_eigen(velocity_meas[0], velocity_meas[1], velocity_meas[2]);
                        flow_vec_eigen /= dt;
                        det_flows.push_back(flow_vec_eigen);
                    }
                }

                std::sort(det_flows.begin(), det_flows.end(), [](const Eigen::Vector3d &e1, const Eigen::Vector3d &e2) {
                    return e1.squaredNorm() < e2.squaredNorm();
                });

                // Compute 'mean' flow from the inner quartile
                const unsigned quartile_size = det_flows.size() / 4;

                Eigen::Vector3d mean_flow;
                mean_flow.setZero();

                int num_samples = 0;
                for (int i = quartile_size; i < det_flows.size() - quartile_size; i++) { // Loop through inner quartile
                    mean_flow += det_flows.at(i);
                    num_samples++;
                }

                if (num_samples<10) {
                    return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                                           std::numeric_limits<double>::quiet_NaN(),
                                           std::numeric_limits<double>::quiet_NaN()
                    );
                }

                return mean_flow / static_cast<double>(num_samples);
            }

            // Implements GPB(1), approximation of a Gaussian mixture with an single Gaussian
            // Basically, colapses Mixture-Of-Gaussians into single one.
            bool ComputePoseCovariance(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                       const Eigen::Vector4d &pose_3d,
                                       const std::vector<int> &indices,
                                       const Eigen::Matrix<double, 3, 4> &P_left,
                                       const Eigen::Matrix<double, 3, 4> &P_right,
                                       Eigen::Matrix3d &covariance_matrix_3d, int min_num_points) {

                Eigen::Matrix3d variance_sum;
                variance_sum.setZero();

                int num_pts = 0;
                for (int i = 0; i < indices.size(); i++) {
                    const auto &p = point_cloud->points.at(indices.at(i));
                    if (!std::isnan(p.x)) {
                        Eigen::Vector3d p_eig = p.getVector3fMap().cast<double>();

                        Eigen::Vector2d diff_vec_2D = Eigen::Vector2d(p_eig[0], p_eig[2])-Eigen::Vector2d(pose_3d[0], pose_3d[2]);
                        if ( diff_vec_2D.norm() > 3.0 )
                            continue;

                        Eigen::Matrix3d cov3d;
                        SUN::utils::Camera::ComputeMeasurementCovariance3d(p_eig, 0.5, P_left, P_right, cov3d);
                        //cov3d += 0.1*Eigen::Matrix3d::Identity(); // Gaussian prior
                        variance_sum += cov3d + (pose_3d.head<3>() - p_eig) *
                                                (pose_3d.head<3>() - p_eig).transpose(); // Outer product
                        num_pts ++;
                    }
                }

                if (num_pts <= min_num_points)
                    return false;

                variance_sum /= (num_pts - 1);
                covariance_matrix_3d = variance_sum;

                return true;
            }


            /**
               * @brief Given image and specific bounding box (format: x y w h), TODO.
               * @param image The RGB image, used for computing the histogram.
               * @param bounding_box_2d The 2d bounding-box, representing the detection, for which we compute the hist.
               * @param num_bins_per_channel Specifies discretization of the color channels (4 is empirically a good pick)
               * @author Aljosa (osep@vision.rwth-aachen.de)
               */
            Eigen::VectorXd ComputeColorHistogramNOMT(const cv::Mat &image, const Eigen::Vector4d &bounding_box_2d, int num_bins_per_channel) {

                // Two layers: 1 + 3x3
                const int hist_size = num_bins_per_channel*num_bins_per_channel*10;
                Eigen::VectorXd histogram_out;
                histogram_out.setZero(hist_size);

                cv::Mat image_LAB;
                cv::cvtColor(image, image_LAB, cv::COLOR_BGR2Lab);

                auto compute_hist_for_block = [&image_LAB](const cv::Rect &crop, int start_index, int num_bins, Eigen::VectorXd &hist) {
                    // Loop through block
                    for (int x=crop.x; x<(crop.x+crop.width); x++) {
                        for (int y=crop.y; y<(crop.y+crop.height); y++) {
                            // Foreach pix., read A, B values
                            auto pix_LAB = image_LAB.at<cv::Vec3b>(y,x);
                            uint8_t A = pix_LAB[1];
                            uint8_t B = pix_LAB[2];

                            //printf("LAB: %i, %i, %i\n", int(pix_LAB[0]), int(pix_LAB[1]), int(pix_LAB[2]) );

//                            // A, B in range [-127, 127], transform to [0, 254]
//                            A += 127.0f;
//                            B += 127.0f;
//
                            // Compute bins
                            int bin_A = static_cast<int>(std::floor( ((float)A / 255.0f)*num_bins ));
                            int bin_B = static_cast<int>(std::floor( ((float)B / 255.0f)*num_bins ));

                            //int linear_index = bin_A * num_bins + bin_B;
                            int linear_index = std::min(bin_A * num_bins + bin_B, (int) hist.size() - start_index); // Alina's fix

                            hist[start_index + linear_index] ++;
                        }
                    }
                };

                // Compute histogram for whole image
                auto det_rect = cv::Rect(bounding_box_2d[0],bounding_box_2d[1],bounding_box_2d[2], bounding_box_2d[3]);
                compute_hist_for_block(det_rect, 0, 4, histogram_out);

                // Compute histogram for 3x3 sub-blocks
                for (int i=0; i<3; i++) {
                    for (int j=0; j<3; j++) {
                        // Compute block indices
                        int block_w = det_rect.width/3;
                        int block_h = det_rect.height/3;
                        int offs_x = det_rect.x + i*block_w;
                        int offs_y = det_rect.y + j*block_h;

                        // Call 'compute_hist_for_block'
                        const int subhist_size = (num_bins_per_channel*num_bins_per_channel);
                        compute_hist_for_block(cv::Rect(offs_x,offs_y,block_w, block_h), subhist_size+(i*3*subhist_size)+(j*subhist_size), 4, histogram_out);
                    }
                }

                double norm_fct = histogram_out.sum();
                histogram_out /= norm_fct;

                // Little unit test
                if (fabs(1.0-histogram_out.sum()) > 0.01) {
                    printf("Detection::ComputeColorHistogramNOMT:Error: Area under the curve does not sum to 1, %d!", (int)histogram_out.sum());
                    assert(false);
                }

                return histogram_out;
            }

            Eigen::Vector4d GetDetectionFootpointFromImageBoundingBox(const SUN::utils::Camera &camera, const Eigen::Vector4d &bounding_box_2d) {
                int footpoint_pix_x = static_cast<int>(bounding_box_2d[0] + (bounding_box_2d[2])/2.0);
                int footpoint_pix_y = static_cast<int>(bounding_box_2d[1] + (bounding_box_2d[3]));

                auto ray = camera.GetRayCameraSpace(footpoint_pix_x, footpoint_pix_y);
                Eigen::Vector4d footpoint;
                footpoint.head<3>() = camera.ground_model()->IntersectRayToGround(ray.origin, ray.direction);
                footpoint[3] = 1.0;

                return footpoint;
            }

        }
    }
}
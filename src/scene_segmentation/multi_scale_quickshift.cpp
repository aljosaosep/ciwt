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
#include "scene_segmentation/multi_scale_quickshift.h"

// segmentation
#include <scene_segmentation/ground_histogram.h>
#include <scene_segmentation/scene_segmentation.h>
#include <scene_segmentation/utils_segmentation.h>

// pcl
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>

// opencv
#include <opencv2/core/eigen.hpp>

// utils
#include "sun_utils/utils_pointcloud.h"
#include "sun_utils/utils_observations.h"
#include "sun_utils/utils_common.h"
#include "sun_utils/utils_filtering.h"


double ComputeBoundingBox2dDensity(const GOT::segmentation::ObjectProposal &object_proposal) {
    const auto &bbox2d = object_proposal.bounding_box_2d();
    return static_cast<double>(object_proposal.pointcloud_indices().size()) /
           static_cast<double>((bbox2d[2] * bbox2d[3]));
}

namespace GOT {
    namespace segmentation {

        namespace proposal_generation {

            std::vector<ObjectProposal>
            ComputeSuppressedMultiScale3DProposals(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr input_cloud,
                                         const SUN::utils::Camera &left_camera, const SUN::utils::Camera &right_camera,
                                                   const po::variables_map &parameter_map) {

                auto proposals_per_scale = ComputeMultiScale3DProposalsQuickShift(input_cloud, left_camera, parameter_map);

                auto scale_overlap_thresh = parameter_map.at("clustering_scale_persistence_overlap_threshold").as<double>();
                auto final_proposal_set = GOT::segmentation::utils::MultiScaleSuppression(proposals_per_scale, scale_overlap_thresh);

                /// Compute covariance matrices
                for (auto &proposal:final_proposal_set) {
                    // TODO ...

                    Eigen::Matrix3d cov_mat;
                    cov_mat.setIdentity();
                    SUN::utils::observations::ComputePoseCovariance(input_cloud, proposal.pos3d(), proposal.pointcloud_indices(),
                                                              left_camera.P().block(0,0,3,4), right_camera.P().block(0,0,3,4),
                                                              cov_mat);
                    proposal.set_pose_covariance_matrix(cov_mat);
                }


                /// Normalize scale-persistance scores
                int num_scale_spaces = parameter_map.at("clustering_num_scale_spaces").as<int>();
                for (auto &proposal:final_proposal_set)
                    proposal.set_score(static_cast<double>(proposal.scale_pairs().size()) / static_cast<double>(num_scale_spaces));

                /// Sort according to the score!
                std::sort(final_proposal_set.begin(), final_proposal_set.end(), [](const GOT::segmentation::ObjectProposal &i, const GOT::segmentation::ObjectProposal &j){ return i.score() > j.score(); });

                return final_proposal_set;
            }

            std::vector <std::vector<ObjectProposal>>
            ComputeMultiScale3DProposalsFloodfill(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr input_cloud,
                                                   const SUN::utils::Camera &camera, const po::variables_map &parameter_map) {

                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_process(new pcl::PointCloud <pcl::PointXYZRGBA>);
                pcl::copyPointCloud(*input_cloud, *cloud_to_process);

                double plane_min_dist = parameter_map.at("filtering_min_distance_to_plane").as<double>();
                double plane_max_dist = parameter_map.at("filtering_max_distance_to_plane").as<double>();
                int min_num_points = parameter_map.at("filtering_min_num_points").as<int>();

                /// Prefilter the point-cloud
                SUN::utils::filter::FilterPointCloudBasedOnDistanceToGroundPlane(cloud_to_process, camera.ground_model(),
                                                                                 plane_min_dist, plane_max_dist, false); // Remove ground-plane points, far-away points


                /// Compute 'ground' density map
                GOT::segmentation::GroundHistogram ground_hist(parameter_map.at("ground_histogram_length").as<double>(),
                                                               parameter_map.at("ground_histogram_depth").as<double>(),
                                                               parameter_map.at("ground_histogram_height").as<double>(),
                                                               parameter_map.at("ground_histogram_rows").as<int>(),
                                                               parameter_map.at("ground_histogram_cols").as<int>());


                //ground_hist.ComputeDensityMap(cloud_to_process, camera, true, 1500, parameter_map.at("ground_histogram_noise_threshold").as<double>());
                ground_hist.ComputeDensityMap(cloud_to_process, camera, false, 1.0, 0.0);

                std::vector<ObjectProposal> proposals_all;
                std::vector <std::vector<ObjectProposal> > proposals_per_scale; // This is a scale-space storage for proposals

                const double intial_radius = parameter_map.at("clustering_initial_kernel_radius").as<double>();
                std::vector<std::pair<float, float> > sigma_inc_pairs = {std::pair<float, float>(intial_radius, intial_radius)};


                std::vector<cv::Mat> scale_space_smooth_maps;
                std::vector<cv::Mat> cc_maps;

                for (const auto &sigma_pair:sigma_inc_pairs) {

                    auto stddev_x = sigma_pair.first;
                    auto stddev_y = sigma_pair.second;

                    cv::Mat prev_smooth_map;
                    int num_scale_spaces = parameter_map.at("clustering_num_scale_spaces").as<int>();
                    for (int i = 0; i < num_scale_spaces; i++) { // PARAM

                        Eigen::MatrixXd convolved_density_map;
                        cv::Mat dst;

                        if (i == 0) {
                            cv::Mat src;
                            cv::eigen2cv(ground_hist.density_map(), src);
                            prev_smooth_map = src.clone();
                            src.release();
                        }

                        // -------- CPU -----------
                        cv::GaussianBlur(prev_smooth_map, dst, cv::Size(0, 0), stddev_x, stddev_y);//1.0, 1.0);
                        // -----------------------

                        cv::cv2eigen(dst, convolved_density_map);

                        /// Run connected-components (optimization; then we only need to segm. further the CCs)
                        Eigen::MatrixXi connected_component_map;
                        std::vector<std::vector<int> > connected_components_to_gp_indices_map;
                        GOT::segmentation::RunConnectedComponents(convolved_density_map, connected_component_map,
                                                                  connected_components_to_gp_indices_map);
                        dst.release();

                        /// Turn clusters into 'proposals'!
                        std::vector<GOT::segmentation::ObjectProposal> proposals_current_scale;
                        for (const auto &cluster:connected_components_to_gp_indices_map) {

                            std::vector<int> pointcloud_inds;
                            Eigen::Vector4d bbox2d;
                            Eigen::VectorXd bbox3d;
                            Eigen::Vector4d gp_bbox2d;

                            std::vector<int> groundplane_inds;
                            for (int gp_index:cluster) {
                                int u = 0, v = 0;
                                SUN::utils::UnravelIndex(gp_index, ground_hist.cols(), &u, &v);
                                const auto &cell_indices = ground_hist.point_index_map()(v, u);
                                if (cell_indices.size() > 1)
                                    groundplane_inds.push_back(gp_index);
                                pointcloud_inds.insert(pointcloud_inds.end(), cell_indices.begin(), cell_indices.end());
                            }

                            GOT::segmentation::ObjectProposal prop;

                            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr obj_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
                            for (auto ind:pointcloud_inds)
                                obj_cloud->points.push_back(input_cloud->at(ind));


                            // TODO: finish this!
                            if (GOT::segmentation::utils::PointsToProposalRepresentation(obj_cloud, camera, bbox2d, bbox3d, min_num_points)) {
                                Eigen::Vector4d gp_bbox = GOT::segmentation::utils::GroundPlaneBoundingBox2D(ground_hist, cluster);

                                prop.init(groundplane_inds, pointcloud_inds, bbox2d, bbox3d, 0, 1.0);
                                prop.set_bounding_box_2d_ground_plane(gp_bbox);
                                prop.add_scale_pair(std::pair<float, float>(stddev_x, stddev_y));

                                /// Coarse geom. filtering
                                // Perform coarse geom. classification (ie. eliminate geom. implausible proposals)
                                // Only want to classify one, but method accepts vector;
                                // Workaround: pass vector, containing a single object
                                std::vector<GOT::segmentation::ObjectProposal> prop_vec = {prop};
                                auto class_vec = GOT::segmentation::utils::CoarseGeometricClassification(prop_vec, camera, min_num_points, 2.0, 0.5);
                                const auto coarse_class = class_vec.at(0);

                                // Turn proposal into a label (for exporting to KITTI format)
                                if (coarse_class == SUN::shared_types::UNKNOWN_TYPE) {
                                    proposals_current_scale.push_back(prop);
                                }
                            }
                        }
                        proposals_per_scale.push_back(proposals_current_scale);

                        // Increase smoothing kernels size for the next iter
                        stddev_x += sigma_pair.first;
                        stddev_y += sigma_pair.second;
                    }
                }

                return proposals_per_scale;
            }



            std::vector <std::vector<ObjectProposal>>
            ComputeMultiScale3DProposalsQuickShift(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr input_cloud,
                                                   const SUN::utils::Camera &camera, const po::variables_map &parameter_map) {

                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_process(new pcl::PointCloud <pcl::PointXYZRGBA>);
                pcl::copyPointCloud(*input_cloud, *cloud_to_process);

                // Idea:
                // 1) Filter the point-cloud
                // 2) Compute the density map
                // 3) Compute proposals for each sigma pair:
                //	a) Convolution
                //	b) QuickShift
                //	c) Clusters -> proposal sets
                // 4) Multi-Scale merging


                double plane_min_dist = parameter_map.at("filtering_min_distance_to_plane").as<double>();
                double plane_max_dist = parameter_map.at("filtering_max_distance_to_plane").as<double>();
                int min_num_points = parameter_map.at("filtering_min_num_points").as<int>();

                /// Prefilter the point-cloud
                SUN::utils::filter::FilterPointCloudBasedOnDistanceToGroundPlane(cloud_to_process, camera.ground_model(),
                                                                                     plane_min_dist, plane_max_dist, false); // Remove ground-plane points, far-away points


                //SUN::utils::filter::FilterPointCloudBasedOnNormalEstimation(cloud_to_process, false); // Normal-based cleaning



                /// Compute 'ground' density map
                GOT::segmentation::GroundHistogram ground_hist(parameter_map.at("ground_histogram_length").as<double>(),
                                                               parameter_map.at("ground_histogram_depth").as<double>(),
                                                               parameter_map.at("ground_histogram_height").as<double>(),
                                                               parameter_map.at("ground_histogram_rows").as<int>(),
                                                               parameter_map.at("ground_histogram_cols").as<int>());


                ground_hist.ComputeDensityMap(cloud_to_process, camera, true, 1500, parameter_map.at("ground_histogram_noise_threshold").as<double>());

                std::vector<ObjectProposal> proposals_all;
                std::vector <std::vector<ObjectProposal> > proposals_per_scale; // This is a scale-space storage for proposals

                const double intial_radius = parameter_map.at("clustering_initial_kernel_radius").as<double>();
                std::vector<std::pair<float, float> > sigma_inc_pairs = {std::pair<float, float>(intial_radius, intial_radius),
                                                                         std::pair<float, float>(intial_radius*0.5, intial_radius),
                                                                         std::pair<float, float>(intial_radius, intial_radius*0.5)};

                /// Multiscale search: get objects candidates at several different scales
                const clock_t clustering_begin_time = clock();
                for (const auto &sigma_pair:sigma_inc_pairs) {

                    auto stddev_x = sigma_pair.first;
                    auto stddev_y = sigma_pair.second;

                    cv::Mat prev_smooth_map;
                    int num_scale_spaces = parameter_map.at("clustering_num_scale_spaces").as<int>();
                    for (int i = 0; i < num_scale_spaces; i++) { // PARAM

                        Eigen::MatrixXd convolved_density_map;
                        cv::Mat dst;

                        if (i == 0) {
                            cv::Mat src;
                            cv::eigen2cv(ground_hist.density_map(), src);
                            prev_smooth_map = src.clone();
                            src.release();
                        }

                        // -------- CPU -----------
                        cv::GaussianBlur(prev_smooth_map, dst, cv::Size(0, 0), stddev_x, stddev_y);//1.0, 1.0);
                        // -----------------------

                        // Enable this line if you want to reuse prev/ convolution result
                        // ie. for incremental filtering
                        // might make everything a tiny bit faster
                        //prev_smooth_map = dst.clone();

                        cv::cv2eigen(dst, convolved_density_map);

                        /// Run connected-components (optimization; then we only need to segm. further the CCs)
                        Eigen::MatrixXi connected_component_map;
                        std::vector <std::vector<int>> connected_components_to_gp_indices_map;
                        GOT::segmentation::RunConnectedComponents(convolved_density_map, connected_component_map,
                                                                  connected_components_to_gp_indices_map);

                        /// Segment connected components further using Quick-shift [Vedadi '08]
                        Eigen::MatrixXi quickshift_segments_map;
                        std::vector <std::vector<int>> quickshift_segments_to_gp_indices_map;
                        GOT::segmentation::QuickShift(convolved_density_map, ground_hist.point_index_map(),
                                                      connected_component_map, connected_components_to_gp_indices_map,
                                                      quickshift_segments_map, quickshift_segments_to_gp_indices_map);

                        dst.release();

                        /// Turn clusters into 'proposals'!
                        std::vector<ObjectProposal> proposals_current_scale;
                        for (const auto &cluster:quickshift_segments_to_gp_indices_map) {

                            std::vector<int> pointcloud_inds;
                            Eigen::Vector4d bbox2d;
                            Eigen::VectorXd bbox3d;
                            Eigen::Vector4d gp_bbox2d;

                            std::vector<int> groundplane_inds;
                            for (int gp_index:cluster) {
                                int u = 0, v = 0;
                                SUN::utils::UnravelIndex(gp_index, ground_hist.cols(), &u, &v);
                                const auto &cell_indices = ground_hist.point_index_map()(v, u);
                                if (cell_indices.size() > 5)
                                    groundplane_inds.push_back(gp_index);
                                pointcloud_inds.insert(pointcloud_inds.end(), cell_indices.begin(), cell_indices.end());
                            }

                            GOT::segmentation::ObjectProposal prop;
                            if (GOT::segmentation::utils::PointsToProposalRepresentation(cloud_to_process, pointcloud_inds, bbox2d, bbox3d, min_num_points)) {
                                Eigen::Vector4d gp_bbox = GOT::segmentation::utils::GroundPlaneBoundingBox2D(ground_hist, cluster);

                                prop.init(groundplane_inds, pointcloud_inds, bbox2d, bbox3d, 0, 1.0);
                                prop.set_bounding_box_2d_ground_plane(gp_bbox);
                                prop.add_scale_pair(std::pair<float, float>(stddev_x, stddev_y));

                                double density_bbox_2d = ComputeBoundingBox2dDensity(prop);
                                double gp_bbox_area = gp_bbox[2] * gp_bbox[3]; //>50

                                double density_thresh = parameter_map.at("filtering_cluster_density_threshold").as<double>();
                                if ((density_bbox_2d > density_thresh))// && (gp_bbox_area>generic_object_proposals_app::filtering_bounding_box_area_threshold))
                                    proposals_current_scale.push_back(prop);
                            }
                        }

                        proposals_per_scale.push_back(proposals_current_scale);

                        // Increase smoothing kernels size for the next iter
                        stddev_x += sigma_pair.first;
                        stddev_y += sigma_pair.second;
                    }
                }

                printf("*** Processing time (multi-scale clustering): %.3f s\r\n", float(clock() - clustering_begin_time) / CLOCKS_PER_SEC);

                return proposals_per_scale;
            }
        }
    }
}
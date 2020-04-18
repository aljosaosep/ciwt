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
#include "utils_filtering.h"

// pcl
//#include <pcl/features/integral_image_normal.h>

// utils
#include "ground_model.h"

namespace SUN {
    namespace utils {
        namespace filter {

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr FilterPointCloudBasedOnRadius(
                    pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr input_cloud, double percentage) {
                Eigen::Vector4f mean;
                pcl::compute3DCentroid(*input_cloud, mean);
                std::map<double, int> dist_map;
                for (int i = 0; i < input_cloud->points.size(); i++) {
                    const pcl::PointXYZRGBA &p = input_cloud->at(i);
                    double norm = (p.getVector4fMap() - mean).norm();
                    dist_map.insert(std::pair<double, int>(norm, i)); // Sorts internally
                }
                int final_index = std::floor((double) dist_map.size() * percentage);
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGBA>);
                int idx = 0;
                for (auto &p:dist_map) {
                    if (idx < final_index) { // Put point below certain radius threshold (determined by percentage of accept points)
                        cloud_out->points.push_back(input_cloud->at(p.second));
                    }
                    idx++;
                }
                return cloud_out;
            }

            pcl::PointIndices FilterPointCloudBasedOnRadius(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                                            const std::vector<int> &indices, double percentage) {
                Eigen::Vector4f mean;
                pcl::PointIndices pcl_inds;
                pcl_inds.indices = indices;
                pcl::compute3DCentroid(*scene_cloud, pcl_inds, mean);
                std::map<double, int> dist_map;
                for (int i = 0; i < indices.size(); i++) {
                    int ind = indices[i];
                    const pcl::PointXYZRGBA &p = scene_cloud->at(ind);
                    double norm = (p.getVector4fMap() - mean).norm();
                    dist_map.insert(std::pair<double, int>(norm, ind)); // Sorts internally
                }
                int final_index = std::floor((double) dist_map.size() * percentage);
                pcl::PointIndices inds_out;
                int idx = 0;
                for (auto &p:dist_map) {
                    if (idx <
                        final_index) { // Put point below certain radius threshold (determined by percentage of accept points)
                        inds_out.indices.push_back(p.second);
                    }
                    idx++;
                }
                return inds_out;
            }

            void FilterPointCloudBasedOnDistanceToGroundPlane(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_be_cleaned,
                                                              std::shared_ptr<SUN::utils::GroundModel> ground_model,
                                                              const double minDistance, const double maxDistance,
                                                              bool only_color_outlier_points) {
                for (auto &p:cloud_to_be_cleaned->points) {
                    //if ( (std::abs(p.y)<minDistance) || (std::abs(p.y)>maxDistance) ) {
                    double dist_to_ground = ground_model->DistanceToGround(p.getVector3fMap().cast<double>());

                    if ( (dist_to_ground<minDistance) || (std::abs(dist_to_ground)>maxDistance) ) {
                        if (only_color_outlier_points) {
                            p.r = static_cast<uint8_t>(255);
                            p.g = static_cast<uint8_t>(0);
                            p.b = static_cast<uint8_t>(0);
                            p.a = static_cast<uint8_t>(255);
                        }
                        else {
                            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                            p.r = p.g = p.b = p.a = std::numeric_limits<uint8_t>::quiet_NaN();
                        }
                    }
                }
            }

//            pcl::PointCloud<pcl::Normal>::Ptr FilterPointCloudBasedOnNormalEstimation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_be_cleaned, bool only_color_outlier_points) {
//                // Estimate normals
//                pcl::PointCloud<pcl::Normal>::Ptr normals_cloud (new pcl::PointCloud<pcl::Normal>);
//                pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
//                ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
//                ne.setMaxDepthChangeFactor(0.02f);
//                ne.setNormalSmoothingSize(10.0f);
//                ne.setInputCloud(cloud_to_be_cleaned);
//                ne.compute(*normals_cloud);
//                normals_cloud->width = cloud_to_be_cleaned->width;
//                normals_cloud->height = cloud_to_be_cleaned->height;
//
//                // Set points, for which normal estimation failed, to NaN
//                for (int i=0; i<cloud_to_be_cleaned->width; i++) {
//                    for (int j=0; j<cloud_to_be_cleaned->height; j++) {
//                        pcl::PointXYZRGBA &p = cloud_to_be_cleaned->at(i,j);
//                        const pcl::Normal &n = normals_cloud->at(i,j);
//                        if (std::isnan (p.x) || std::isnan (p.y) || std::isnan (p.z))
//                            continue;
//                        if (std::isnan (n.normal_x) || std::isnan (n.normal_y) || std::isnan (n.normal_z)) {
//                            if (only_color_outlier_points) {
//                                // p.rgb = GOT::utils::pointcloud::PackRgbValuesToUint32(255, 0,0 );
//                                p.r = static_cast<uint8_t>(255);
//                                p.g = static_cast<uint8_t>(0);
//                                p.b = static_cast<uint8_t>(0);
//                                p.a = static_cast<uint8_t>(255);
//                            }
//                            else {
//                                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
//                                p.r = p.g = p.b = p.a = std::numeric_limits<uint8_t>::quiet_NaN();
//                            }
//                        }
//                    }
//                }
//
//                return normals_cloud;
//            }
        }
    }
}
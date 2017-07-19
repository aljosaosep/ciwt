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

#include <scene_segmentation/ground_histogram.h>

// pcl
#include <pcl/common/io.h>

// Utils
#include "camera.h"
#include "utils_io.h"
#include "utils_bounding_box.h"
#include "ground_model.h"

namespace GOT {
    namespace segmentation {

        GroundHistogram::GroundHistogram() {
            length_ = depth_ = height_ = rows_ = cols_ = 0.0;
        }

        GroundHistogram::GroundHistogram(double length, double depth, double height, int rows, int cols) {
            length_ = length;
            depth_ = depth;
            height_ = height;
            rows_ = rows;
            cols_ = cols;
        }

        void GroundHistogram::ComputeDensityMap(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr input_cloud, const SUN::utils::Camera &camera, bool distance_compensate, /*bool is_organized_cloud,*/ const double normalization_factor, const double threshold) {
            // Init underlying data structure.
            density_map_.resize(rows_, cols_);
            density_map_.fill(0.0);

            point_index_map_.resize(rows_, cols_);

            pcl::PointCloud<pcl::PointXYZRGBA> tmp_cloud;
            pcl::copyPointCloud(*input_cloud, tmp_cloud);

            // Project all points to ground-plane!
            for (auto &p:tmp_cloud.points) {
                Eigen::Vector3d p_proj = camera.ground_model()->ProjectPointToGround(p.getVector3fMap().cast<double>());
                p.x = p_proj[0];
                p.y = p_proj[1];
                p.z = p_proj[2];
            }

            // Transform point cloud to [0,1]^3.
            Eigen::Vector4f min_pt, max_pt;
            pcl::getMinMax3D(tmp_cloud, min_pt, max_pt);
            for (pcl::PointXYZRGBA &p:tmp_cloud.points) {
                if (!std::isnan(p.x)) {
                    p.getVector4fMap() += Eigen::Vector4f(this->length_/2.0, 0.0, 0.0 ,1.0);
                    p.getVector4fMap()[3] = 1.0;
                }
            }

            // Normalize and reject points that fall out of range.
            for (pcl::PointXYZRGBA &p:tmp_cloud.points) {
                if (!std::isnan(p.x)) {
                    p.x /= this->length_;
                    p.y /= this->height_;
                    p.z /= this->depth_;

                    // Did point fall out of range? If yes, NaN-ify it!
                    if ( (p.x > 1.0) || (p.y > 1.0) || (p.z > 1.0)
                         || (p.x < 0.0) || (p.y < 0.0) || (p.z < 0.0) ) {
                        p.x = std::numeric_limits<float>::quiet_NaN();
                        p.y = std::numeric_limits<float>::quiet_NaN();
                        p.z = std::numeric_limits<float>::quiet_NaN();
                        p.r = std::numeric_limits<uint8_t>::quiet_NaN();
                        p.g = std::numeric_limits<uint8_t>::quiet_NaN();
                        p.b = std::numeric_limits<uint8_t>::quiet_NaN();
                        p.a = std::numeric_limits<uint8_t>::quiet_NaN();
                    }
                }
            }

            const double log_2 = std::log(2);

            // Bin the point cloud to the 2D grid (loose Y-coord!).
            int point_index = 0;
            for (const auto &p:tmp_cloud) {
                if (!std::isnan(p.x)) {
                    const int bin_x =  std::floor(p.x*(this->cols_-1));
                    const int bin_z = std::floor(p.z*(this->rows_-1));
                    //const int bin_x =  std::floor(p.x*(this->cols_));
                    //const int bin_z = std::floor(p.z*(this->rows_));

                    const double input_cloud_z = input_cloud->at(point_index).z; // This is dist. from camera
                    if (distance_compensate)
                        this->density_map_(bin_z, bin_x) += std::max(1.0, input_cloud_z*(std::log(input_cloud_z) / log_2 ));
                    else
                        this->density_map_(bin_z, bin_x) += 1.0;

                    this->point_index_map_(bin_z, bin_x).push_back(point_index);

                    /*if (is_organized_cloud)
                        this->point_index_map_(bin_z, bin_x).push_back(point_index);
                    else {
                        // Non-org. cloud: project to image and compute (linear) index in image plane
                        Eigen::Vector3i p_proj = camera.CameraToImage(Eigen::Vector4d(p.x, p.y, p.z, 1.0));
                        int idx;
                        SUN::utils::geometry::RavelIndex(p_proj[0], p_proj[1], camera.width(), &idx);
                        this->point_index_map_(bin_z, bin_x).push_back(idx);
                    }*/
                }
                point_index ++;
            }


            density_map_ /= normalization_factor;

            // Threshold the density map
            for (int i=0; i<density_map_.size(); i++) {
                if (density_map_(i) > 1.0)
                    density_map_(i) = 1.0;

                if (density_map_(i) < threshold)
                    density_map_(i) = 0.0;
            }

            this->ComputeHeightMaps(input_cloud, camera);
        }

        bool GroundHistogram::GetBinFrom3dPoint(const SUN::utils::Camera &camera, const Eigen::Vector3d &point_3d, int &row, int &col) const {
            row = col = -1;

            auto p = camera.ground_model()->ProjectPointToGround(point_3d);

            p += Eigen::Vector3d(this->length_/2.0, 0.0, 0.0 );
            p[0] /= this->length_;
            p[1] /= this->height_;
            p[2] /= this->depth_;

            // Did point fall out of range? If yes, NaN-ify it!
            if ( (p[0] > 1.0) || (p[2] > 1.0) || (p[0] < 0.0) || (p[2] < 0.0) ) {
                return false;
            }

            col =  std::floor(p[0]*(this->cols_-1));
            row = std::floor(p[2]*(this->rows_-1));
            return true;
        }

        // Setters / getters
        double GroundHistogram::rows() const {
            return this->rows_;
        }

        double GroundHistogram::cols() const {
            return this->cols_;
        }

        const Eigen::MatrixXd& GroundHistogram::density_map() const {
            return this->density_map_;
        }

        const Eigen::Matrix<std::vector<int>, Eigen::Dynamic, Eigen::Dynamic>& GroundHistogram::point_index_map() const {
            return this->point_index_map_;
        }

        void GroundHistogram::ComputeHeightMaps(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr input_cloud,
                                                const SUN::utils::Camera &camera) {
            // Do:
            // For each cell:
            //  - Access point set
            //  - Sort by Y-coord
            //  - Height ~ take highest, reject % of pts
            this->max_height_map_.setZero(point_index_map_.rows(), point_index_map_.cols());
            this->min_height_map_.setZero(point_index_map_.rows(), point_index_map_.cols());

            for (int i=0; i<this->point_index_map_.rows(); i++) {
                for (int j=0; j<this->point_index_map_.cols(); j++) {
                    const std::vector<int> &inds = point_index_map_(i,j);
                    std::vector<double> heights;
                    heights.reserve(inds.size());
                    for (auto ind:inds) {
                        const auto p = input_cloud->at(ind);

                        if (!std::isnan(p.x)) {
                            double dist = camera.ground_model()->DistanceToGround(p.getVector3fMap().cast<double>());
                            heights.push_back(dist);
                        }
                    }

                    if (heights.size()>3) {
                        std::sort(heights.begin(), heights.end());
                        int height_index = std::floor(static_cast<double>(heights.size()) * 0.95);
                        max_height_map_(i, j) = heights.at(height_index);
                        min_height_map_(i, j) = heights.at(heights.size()-height_index);
                    }
                }

            }
        }

        const Eigen::MatrixXd& GroundHistogram::max_height_map() const {
            return max_height_map_;
        }

        const Eigen::MatrixXd &GroundHistogram::min_height_map() const {
            return min_height_map_;
        }
    }
}

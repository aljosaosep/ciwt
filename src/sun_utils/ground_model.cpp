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

#include "ground_model.h"

// pcl
#include <pcl/common/centroid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace SUN {
    namespace utils {

        double PlanarGroundModel::DistanceToGround(const Eigen::Vector3d &p) const {
            double a = plane_params_[0];
            double b = plane_params_[1];
            double c = plane_params_[2];
            double d = plane_params_[3];

            double x = p[0];
            double y = p[1];
            double z = p[2];
            double dist = a * x + b * y + c * z + d; //(std::abs(a * x + b * y + c * z + d)) /
                          // (std::sqrt(a * a + b * b + c * c)); // Dist. point to the plane. Distance in the plane normal direction.
            return dist;
        }

        Eigen::Vector3d PlanarGroundModel::ProjectPointToGround(const Eigen::Vector3d &p) const {
            double a = plane_params_[0];
            double b = plane_params_[1];
            double c = plane_params_[2];

            Eigen::Vector3d p_proj = p;
            p_proj -= DistanceToGround(p)*Eigen::Vector3d(a,b,c); // Substract plane_normal*dist_to_plane, and we have a projected point!

            return p_proj;
        }

        Eigen::Vector3d PlanarGroundModel::IntersectRayToGround(const Eigen::Vector3d &ray_origin, const Eigen::Vector3d &ray_direction) const {
            Eigen::Vector3d ground_plane_normal = plane_params_.head(3);
            double d = plane_params_[3];
            double nom = (ray_origin.dot(ground_plane_normal) + d);
            double denom = ray_direction.dot(ground_plane_normal);
            //double t = - (ray.origin.dot(ground_plane_normal) + d) / (ray.direction.dot(ground_plane_normal));
            double t = - (nom / denom);

            if (t<0.000001)
                return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

            return ray_origin + t * ray_direction;
        }

        Eigen::Vector4d PlanarGroundModel::FitPlaneToInliersLeastSquares(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &scene_cloud,
                                            const std::vector<int> &indices) {

            Eigen::MatrixXd inlier_matrix;
            inlier_matrix.setZero(indices.size(), 3);

            pcl::PointIndices point_inds;
            point_inds.indices = indices;
            Eigen::Vector4d centroid;
            pcl::compute3DCentroid(*scene_cloud, point_inds, centroid);

            int row_index = 0;
            for (auto ind:indices) {
                const auto &pt = scene_cloud->at(ind);
                const Eigen::Vector3d p_eig(pt.x-centroid[0], pt.y-centroid[1], pt.z-centroid[2]);

                inlier_matrix.row(row_index) = p_eig;
                row_index ++;
            }

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(inlier_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::Vector3d sol = svd.matrixV().col(2); // This is solution to homogeneous nonlin. sys

            double a,b,c,d;
            a = sol[0];
            b = sol[1];
            c = sol[2];
            d = -(centroid[0]*a + centroid[1]*b + c*centroid[2]);

            // Postprocessing
            Eigen::Vector3d n(a,b,c);
            Eigen::Vector4d plane(a,b,c,d);
            plane /= n.norm();

            if (plane[1]>0)
            plane *= -1.0;

            a = plane[0];
            b = plane[1];
            c = plane[2];
            d = plane[3];

            return Eigen::Vector4d(a,b,c,d);
        }

        void PlanarGroundModel::FitModel(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud_in, double height_threshold) {

            const double lateral_threshold = 15; //8;
            const double vertical_threshold = 1.0;

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(
                    new pcl::PointCloud<pcl::PointXYZRGBA>);

            for (int i = 0; i < point_cloud_in->size(); ++i) {
                double y = point_cloud_in->points[i].y;
                double x = point_cloud_in->points[i].x;
                if ( (y > vertical_threshold) && (x < lateral_threshold) && (x > -lateral_threshold) ) { // 1.4
                    cloud->points.push_back(point_cloud_in->points[i]);
                }
            }

            cloud->height = 1;
            cloud->width = (int)cloud->points.size();

            // Make sure that we actually have some points.
            if (cloud->size() < 1000) {
                std::cout << "WARNING: RectifyPointCloud, cloud->size() < 1000 !!! " << std::endl;
                plane_params_ = Eigen::Vector4d(0.0,0.0,0.0,0.0);
            }
            else {
                // PCL RANSAC init.
                pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
                seg.setOptimizeCoefficients(true);
                seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
                seg.setMethodType(pcl::SAC_RANSAC);

                // Give a prior to which plane should be found and limit what deviation is accepted.
                seg.setAxis(Eigen::Vector3f(0.0f, -1.0f, 0.0f));
                seg.setEpsAngle((20 * 3.14) / 180);
                seg.setDistanceThreshold(0.1);
                seg.setMaxIterations(3000);

                // Fit the plane.
                seg.setInputCloud(cloud);
                pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                seg.segment(*inliers, *coefficients);

                if (inliers->indices.size() == 0)
                    throw std::runtime_error("Could not estimate a planar model for the given dataset.");

                Eigen::Vector4d plane_params = Eigen::Vector4d(coefficients->values[0],
                                                               coefficients->values[1],
                                                               coefficients->values[2],
                                                               coefficients->values[3]);

                Eigen::Vector4d refined_plane_params = FitPlaneToInliersLeastSquares(cloud, inliers->indices);
                plane_params = refined_plane_params;

                this->plane_params_ = plane_params;
            }
        }

        Eigen::Vector3d PlanarGroundModel::Normal(const Eigen::Vector3d &point_3d) const {
            // For plane repr. it's easy -- normal is same everywhere.
            // We can ignore the arg!
            double a = plane_params_[0];
            double b = plane_params_[1];
            double c = plane_params_[2];
            return Eigen::Vector3d(a,b,c);
        }

        void PlanarGroundModel::set_plane_params(const Eigen::Vector4d &plane_params) {
            plane_params_ = plane_params;
        }

        const Eigen::Vector4d PlanarGroundModel::plane_params() const {
            return plane_params_;
        }
    }
}

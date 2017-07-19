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

#ifndef GOT_SCENE_SEGMENTATION_UTILS
#define GOT_SCENE_SEGMENTATION_UTILS

// OpenCV
#include <opencv2/core/core.hpp>

// boost
#include <boost/program_options.hpp>

// Segmentation
#include <scene_segmentation/object_proposal.h>
#include <scene_segmentation/ground_histogram.h>

// Utils
#include "sun_utils/utils_kitti.h"

// Forward declarations
namespace SUN { class Calibration; }

namespace po = boost::program_options;

namespace GOT {
    namespace segmentation {

        namespace utils {

            /**
             * @brief Filter super-implausible proposals based on physical dimensions and min. no. point support.
             * @param object_proposals_in The 'raw' proposal set.
             * @param camera The camera for current frame.
             * @param center_height_threshold Don't take proposals with centroid above this thresh. We don't track planes.
             * @param min_num_ponts_threshold Don't take regions with less-than-this no. points.
             * @param min_height_threshold Don't take proposals, smaller (height) than this.
             * @return Filtered proposal set.
            */
            std::vector<ObjectProposal> FilterProposals(const std::vector<ObjectProposal> &object_proposals_all, const SUN::utils::Camera &camera,
                                                        po::variables_map &parameter_map);




            /**
             * @brief Coarsly classify proposals based on scene geom. (UNKNOWN, NOISE)
             * @param proposals The 'raw' proposal set.
             * @param camera The camera for current frame.
             * @param min_num_ponts_threshold Don't take regions with less-than-this no. points.
             * @param center_height_threshold Mark as noise, proposals with bottom-coord above this thresh
             * @param min_height_threshold Mark as noise everything with height below this thresh.
             * @return Filtered proposal set.
            */
            std::vector<SUN::shared_types::CategoryTypeKITTI> CoarseGeometricClassification(const std::vector<GOT::segmentation::ObjectProposal> &proposals, const SUN::utils::Camera &camera,
                                                                                       double min_num_points_threshold, double center_height_threshold, double min_height_threshold);


            /**
             * @brief Turns a set of 3D points to 2D, 3D bounding-boxes. As input, seg. mask is assumed (indices).
            */
            bool PointsToProposalRepresentation(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                                const std::vector<int> &object_indices,
                                                Eigen::Vector4d &bbox2d,
                                                Eigen::VectorXd &bbox3d,
                                                int min_cluster_size=100);

            /**
             * @brief Turns a set of 3D points to 2D, 3D bounding-boxes. As input, 3D point segment is assumed.
            */
            bool PointsToProposalRepresentation(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr segment_cloud,
                                                const SUN::utils::Camera &camera,
                                                Eigen::Vector4d &bbox2d,
                                                Eigen::VectorXd &bbox3d,
                                                int min_cluster_size=100);

            /**
             * @brief Computes ground-plane bounding-box.
            */
            Eigen::Vector4d GroundPlaneBoundingBox2D(const GOT::segmentation::GroundHistogram &ground_hist,
                                                     const std::vector<int> gp_indices, int num_points_threshold=3);

            /**
             * @brief Turns proposals to labels - useful for evaluation.
            */
            void ProposalsSetToLabels(int frame, const std::vector<ObjectProposal> &proposals, std::vector<SUN::utils::KITTI::TrackingLabel> &labels);

            /**
             * @brief Merges similar-sized proposals in neighbouring scales.
            */
            std::vector<ObjectProposal> MultiScaleSuppression(const std::vector<std::vector<ObjectProposal>> &proposals_per_scale, double IOU_threshold);


            // I/O
            bool SaveObjectProposals(const char *filename, const std::vector<ObjectProposal> &proposals);
            bool LoadObjectProposals(const char *filename, std::vector<ObjectProposal> &proposals);
        }
    }
}

#endif

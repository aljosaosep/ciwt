/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Dennis Mitzel (osep, mitzel -at- vision.rwth-aachen.de)

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

#ifndef GOT_SCENE_SEGMENTATION_NEW_H
#define GOT_SCENE_SEGMENTATION_NEW_H

#include <scene_segmentation/ground_histogram.h>
#include <scene_segmentation/object_proposal.h>


namespace GOT {
    namespace segmentation {
        int RunConnectedComponents(const Eigen::MatrixXd &density_map,
                                   Eigen::MatrixXi &connected_component_map,
                                   std::vector <std::vector<int>> &connected_components_to_gp_indices_map);

        int QuickShift(const Eigen::MatrixXd &density_map,
                       const Eigen::Matrix<std::vector<int>, Eigen::Dynamic, Eigen::Dynamic>& point_index_map,
                       const Eigen::MatrixXi &connected_component_map,
                       const std::vector <std::vector<int>> &connected_components_to_gp_indices_map,
                       Eigen::MatrixXi &quickshift_segments_map,
                       std::vector <std::vector<int>> &quickshift_segments_to_gp_indices_map);


        bool PointsToProposalRepresentation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_cloud, const std::vector<int> &object_indices, Eigen::Vector4d &bbox2d, Eigen::VectorXd &bbox3d);
    }
}


#endif //GOT_SCENE_SEGMENTATION_NEW_H

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

#include <scene_segmentation/utils_segmentation.h>

//// boost
//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/binary_iarchive.hpp>

// utils
#include "utils_observations.h"
#include "camera.h"
#include "utils_kitti.h"
#include "ground_model.h"
#include "utils_common.h"
#include "utils_bounding_box.h"

namespace GOT {
    namespace segmentation {
        namespace utils {

            std::vector<ObjectProposal> FilterProposals(const std::vector<ObjectProposal> &object_proposals_in,
                                                                    const SUN::utils::Camera &camera,
                                                                    po::variables_map &parameter_map) {

                assert(parameter_map.count("proposals_max_height_offset"));
                assert(parameter_map.count("proposals_min_points"));
                assert(parameter_map.count("proposals_min_height"));

                double center_height_threshold = parameter_map.at("proposals_max_height_offset").as<double>();
                int min_num_points_threshold = parameter_map.at("proposals_min_points").as<int>();
                double min_height_threshold = parameter_map.at("proposals_min_height").as<double>();

                std::vector<ObjectProposal> proposals_filtered;
                proposals_filtered.reserve(object_proposals_in.size());

                for (const auto proposal:object_proposals_in) {
                    bool accept_proposal = true;
                    if (proposal.pointcloud_indices().size()<min_num_points_threshold)
                        accept_proposal = false;
                    if (camera.ground_model()->DistanceToGround(proposal.pos3d().head<3>()) > center_height_threshold)
                        accept_proposal = false;
                    if (proposal.bounding_box_3d()[4] < min_height_threshold)
                        accept_proposal = false;
                    if (accept_proposal)
                        proposals_filtered.push_back(proposal);
                }

                return proposals_filtered;
            }

            bool PointsToProposalRepresentation(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                                const std::vector<int> &object_indices,
                                                Eigen::Vector4d &bbox2d,
                                                Eigen::VectorXd &bbox3d, int min_cluster_size) {

                const double percentage = 1.0; // 0.95; // Percentage of the points "we trust".
                if (object_indices.size() >= min_cluster_size) {
                    bbox2d = SUN::utils::bbox::BoundingBox2d(scene_cloud, object_indices, percentage);
                    bbox3d = SUN::utils::bbox::BoundingBox3d(scene_cloud, object_indices, percentage);
                    return true;
                }

                return false;
            }

            bool PointsToProposalRepresentation(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr segment_cloud,
                                                const SUN::utils::Camera &camera,
                                                Eigen::Vector4d &bbox2d,
                                                Eigen::VectorXd &bbox3d, int min_cluster_size) {

                const double percentage = 1.0; // 0.95; // Percentage of the points "we trust".
                if (segment_cloud->size() >= min_cluster_size) {
                    // non-organized
                    bbox2d = SUN::utils::bbox::BoundingBox2d(segment_cloud, camera, percentage);
                    bbox3d = SUN::utils::bbox::BoundingBox3d(segment_cloud, percentage);
                    return true;
                }

                return false;
            }

            Eigen::Vector4d GroundPlaneBoundingBox2D(const GOT::segmentation::GroundHistogram &ground_hist,
                                                     const std::vector<int> gp_indices, int num_points_threshold) {
                int min_u = 1e5, max_u = -1e5, min_v = 1e5, max_v = -1e5;
                for (auto ind:gp_indices) {
                    int u = 0, v = 0;
                    SUN::utils::UnravelIndex(ind, ground_hist.cols(), &u, &v);
                    if (u>0&&v>0&&u<ground_hist.cols()&&v<ground_hist.rows()) {
                        // Each 'dense' cell pushes limit
                        if (ground_hist.point_index_map()(v, u).size() > num_points_threshold) {
                            if (u < min_u)
                                min_u = u;
                            if (u > max_u)
                                max_u = u;
                            if (v < min_v)
                                min_v = v;
                            if (v > max_v)
                                max_v = v;
                        }
                    }
                }
                return Eigen::Vector4d(min_u, min_v, max_u - min_u, max_v - min_v);
            }

            std::vector<SUN::shared_types::CategoryTypeKITTI> CoarseGeometricClassification(const std::vector<GOT::segmentation::ObjectProposal> &proposals, const SUN::utils::Camera &camera,
                                                                                       double min_num_points_threshold, double center_height_threshold, double min_height_threshold) {
                std::vector<SUN::shared_types::CategoryTypeKITTI> categ_types(proposals.size());
                for (int i=0; i<proposals.size(); i++) {
                    const auto &prop = proposals.at(i);
                    // Compute the "bottom point" of the proposal (centroid minus half-height)
                    Eigen::Vector3d proposal_centroid = prop.bounding_box_3d().head<3>();
                    Eigen::Vector3d proposal_bottom_point = proposal_centroid;
                    proposal_bottom_point -= camera.ground_model()->Normal(proposal_bottom_point) * (prop.bounding_box_3d()[4]/2.0);
                    Eigen::Vector3d proposal_top_point = proposal_centroid+camera.ground_model()->Normal(proposal_centroid) * (prop.bounding_box_3d()[4]/2.0);
                    if (prop.pointcloud_indices().size()<min_num_points_threshold)
                        categ_types.at(i) = SUN::shared_types::NOISE;
                    else if (std::fabs(camera.ground_model()->DistanceToGround(proposal_bottom_point)) > center_height_threshold)
                        categ_types.at(i) = SUN::shared_types::NOISE;
                    else if (std::fabs(camera.ground_model()->DistanceToGround(proposal_top_point)) >  3.5)
                        categ_types.at(i) = SUN::shared_types::NOISE;
                    else if (std::fabs(camera.ground_model()->DistanceToGround(proposal_top_point)) < min_height_threshold)
                        categ_types.at(i) = SUN::shared_types::NOISE;
                    else
                        categ_types.at(i) = SUN::shared_types::UNKNOWN_TYPE;
                }
                return categ_types;
            }

            std::vector<ObjectProposal> MultiScaleSuppression(
                    const std::vector<std::vector<ObjectProposal> > &proposals_per_scale, double IOU_threshold) {

                std::vector<ObjectProposal> accepted_proposals;
                for (const auto &current_scale_proposals:proposals_per_scale) {
                    // Loop through current-scale proposals. If you find IOU overlap with existing, merge the two. Otherwise, push to list.

                    for (const auto &curr_scale_proposal:current_scale_proposals) {
                        bool overlap_found = false;
                        for (auto &accepted_proposal:accepted_proposals) {
                            if ((SUN::utils::bbox::IntersectionOverUnion2d(curr_scale_proposal.bounding_box_2d_ground_plane(),
                                                                              accepted_proposal.bounding_box_2d_ground_plane()) > IOU_threshold) &&
                                            // NEW: require both, GP and IMG domain bboxes to be consistent
                                    (SUN::utils::bbox::IntersectionOverUnion2d(curr_scale_proposal.bounding_box_2d(),
                                                                                   accepted_proposal.bounding_box_2d()) > IOU_threshold))



                            {
                                overlap_found = true;

                                accepted_proposal.add_scale_pairs(curr_scale_proposal.scale_pairs());

                                // -------------
                                Eigen::MatrixXd merged_bbox_2d =
                                        (accepted_proposal.bounding_box_2d() + curr_scale_proposal.bounding_box_2d()) / 2.0;
                                //Eigen::MatrixXd merged_bbox_3d = (accepted_proposal.bounding_box_3d() + curr_scale_proposal.bounding_box_3d())/2.0;
                                Eigen::Vector4d merged_pos_3d = (accepted_proposal.pos3d() + curr_scale_proposal.pos3d()) / 2.0;
                                Eigen::Vector4d merged_bbox_2d_gp = (accepted_proposal.bounding_box_2d_ground_plane() +
                                                                     curr_scale_proposal.bounding_box_2d_ground_plane()) / 2.0;

                                // --------- point-cloud inds ---------------
                                // Union
                                std::vector<int> merged_inds = accepted_proposal.pointcloud_indices();
                                std::vector<int> new_inds = curr_scale_proposal.pointcloud_indices();
                                merged_inds.insert(merged_inds.end(), new_inds.begin(), new_inds.end());

                                // Make unique
                                std::sort(merged_inds.begin(), merged_inds.end());
                                merged_inds.erase(std::unique(merged_inds.begin(), merged_inds.end()), merged_inds.end());
                                // -------------------------------------------

                                // --------- GP inds ---------------
                                // Union
                                std::vector<int> merged_inds_gp = accepted_proposal.ground_plane_indices();
                                std::vector<int> new_inds_gp = curr_scale_proposal.ground_plane_indices();
                                merged_inds_gp.insert(merged_inds_gp.end(), new_inds_gp.begin(), new_inds_gp.end());

                                // Make unique
                                std::sort(merged_inds_gp.begin(), merged_inds_gp.end());
                                merged_inds_gp.erase(std::unique(merged_inds_gp.begin(), merged_inds_gp.end()), merged_inds_gp.end());
                                // -------------------------------------------

                                // !! WARNING !! : 2D, 3D bboxes do not get updated !!
                                accepted_proposal.set_pointcloud_indices(merged_inds);
                                accepted_proposal.set_groundplane_indices(merged_inds_gp);
                                //accepted_proposal.set_bounding_box_2d_ground_plane(merged_bbox_2d);
                                accepted_proposal.set_bounding_box_2d_ground_plane(merged_bbox_2d_gp);
                                //accepted_proposal.set_bounding_box_3d(merged_bbox_3d);
                                accepted_proposal.set_pos3d(merged_pos_3d);
                                // -----------------

                                break;
                            }
                        }

                        if (!overlap_found)
                            accepted_proposals.push_back(curr_scale_proposal);
                    }
                }

                return accepted_proposals;
            }


            void ProposalsSetToLabels(int frame, const std::vector<ObjectProposal> &proposals, std::vector<SUN::utils::KITTI::TrackingLabel> &labels) {
                for(const auto &proposal:proposals) {

                    SUN::utils::KITTI::TrackingLabel label;
                    label.frame = frame;
                    label.trackId = -1;
                    label.type = SUN::shared_types::UNKNOWN_TYPE; //SUN::kitti::UNKNOWN_TYPE;
                    label.truncated = -1;
                    label.occluded = SUN::utils::KITTI::NOT_SPECIFIED;
                    label.alpha = -1;

                    label.boundingBox2D[0] = proposal.bounding_box_2d()[0];
                    label.boundingBox2D[1] = proposal.bounding_box_2d()[1];
                    label.boundingBox2D[2] = proposal.bounding_box_2d()[2] + proposal.bounding_box_2d()[0];
                    label.boundingBox2D[3] = proposal.bounding_box_2d()[3] + proposal.bounding_box_2d()[1];

                    // This should be correct even though index assignment may look incorrect.
                    label.dimensions[0] = proposal.bounding_box_3d()[4];//-1;
                    label.dimensions[1] = proposal.bounding_box_3d()[3]; // -1;
                    label.dimensions[2] = proposal.bounding_box_3d()[5]; // -1;

                    label.location[0] = proposal.pos3d()[0];
                    label.location[1] = proposal.pos3d()[1];
                    label.location[2] = proposal.pos3d()[2];

                    label.rotationY = -10;
                    label.score = proposal.score();

                    labels.push_back(label);
                }
            }

            bool SaveObjectProposals(const char *filename, const std::vector<ObjectProposal> &proposals) {
//                std::ofstream ofs(filename);
//                if (!ofs.is_open())
//                    return false;
//                //boost::archive::text_oarchive oa(ofs);
//                boost::archive::binary_oarchive oa(ofs);
//                int num_proposals = proposals.size();
//                oa << num_proposals;
//                for (const auto &proposal:proposals) {
//                    oa << proposal;
//                }
//                ofs.close();
//
//                return true;
            }

            bool LoadObjectProposals(const char *filename, std::vector<ObjectProposal> &proposals) {
//                proposals.clear();
//                std::ifstream ifs(filename);
//                if (!ifs.is_open())
//                    return false;
//                boost::archive::binary_iarchive ia(ifs); //text_iarchive ia(ifs);
//                int num_proposals=0;
//                ia >> num_proposals;
//                for (int i=0; i<num_proposals; i++) {
//                    GOT::segmentation::ObjectProposal proposal;
//                    ia >> proposal;
//                    proposals.push_back(proposal);
//                }
//                return true;
            }

            std::vector<int> ComputeGroundPlaneIndicesFromPointCloudIndices(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud, const SUN::utils::Camera &camera, const std::vector<int> &pointcloud_inds, const GOT::segmentation::GroundHistogram &ground_hist) {
                std::vector<int> gp_inds;
                for (auto pc_ind:pointcloud_inds) {
                    const auto &p = scene_cloud->at(pc_ind);
                    if (!std::isnan(p.x)) {
                        int row, col;

                        Eigen::Vector3d p_mapped = p.getVector3fMap().cast<double>();
                        bool is_in = ground_hist.GetBinFrom3dPoint(camera, p_mapped, row, col);

                        if (is_in) {
                            int gp_ind;
                            SUN::utils::RavelIndex(col, row, ground_hist.cols(), &gp_ind);
                            gp_inds.push_back(gp_ind);
                        }
                    }
                }

                return gp_inds;
            }

        }
    }
}

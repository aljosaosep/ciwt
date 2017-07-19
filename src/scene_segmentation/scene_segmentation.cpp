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

#include <scene_segmentation/scene_segmentation.h>
#include <scene_segmentation/utils_segmentation.h>

#include <connected_components/CC.h>

#include "sun_utils/utils_common.h"

namespace GOT {
    namespace segmentation {


        int RunConnectedComponents(const Eigen::MatrixXd &density_map,
                                   Eigen::MatrixXi &connected_component_map,
                                   std::vector <std::vector<int>> &connected_components_to_gp_indices_map) {
            ConnectedComponents::CC *ccl = new ConnectedComponents::CC();
            ccl->InitConfig(10); // regions below this threshold are deleted
            ccl->Clear(); // if you are using ccl in a loop, to avoid memory leakages call this function

            // Eigen <-> raw data interface
            // Map data from Eigen storage to C-raw storage, required to interface density map with CC class interface
            // double *raw_input=nullptr;
            // Eigen::Map<Eigen::MatrixXd>(raw_input, rows_, cols_) = this->density_map_;
            // Eigen map does not work for some reason, so I had to copy data in row-order manually...
            const int rows = density_map.rows();
            const int cols = density_map.cols();
            //const auto &density_map = density_map.density_map();
            double *raw_input = new double[rows * cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    const double val = density_map(i, j);
                    raw_input[i * (cols) + j] = val * 255;
                }
            }

            // Process connected components
            ccl->SetMask(raw_input, cols, rows);
            ccl->Process();
            int num_ccs = ccl->m_ObjectNumber; // number of objects found
            int *output_connected_components = ccl->GetOutput();

            // Map CC array->Eigen
            Eigen::Map <Eigen::MatrixXi> cc_map(output_connected_components, cols, rows);
            connected_component_map = cc_map.transpose();

            // Clear memory
            delete[] raw_input;
            ccl->Clear();
            delete ccl;

            // Compute connected components maps. Those are vectors of size equal to number of connected components.
            // Each entry corresponds to connected component index data.
            // E.g. indices of connected component with id 0 are accessible via:
            // auto &indices_of_cc_0 = this->this->connected_components_to_gp_indices_map_->at(0);
            // indices_of_cc_0 will now store an std::vector of linear indices.
            connected_components_to_gp_indices_map.clear();
            connected_components_to_gp_indices_map.resize(num_ccs);
            //connected_components_to_pointcloud_indices_map_.resize(num_ccs);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    int cc_id =
                            connected_component_map(i, j) - 1; // Because CC id's go from 1...num_ccs, 0 is 'background'
                    int ind = -1;
                    if (cc_id > -1) {
                        // Convert groundplane coordinates to indices and push to cc vector
                        SUN::utils::RavelIndex(j, i, cols, &ind);// j,i or i,j?
                        connected_components_to_gp_indices_map.at(cc_id).push_back(ind);
                    }
                }
            }
            return num_ccs;
        }

        int QuickShift(const Eigen::MatrixXd &density_map,
                       const Eigen::Matrix<std::vector<int>, Eigen::Dynamic, Eigen::Dynamic>& point_index_map,
                       const Eigen::MatrixXi &connected_component_map,
                       const std::vector <std::vector<int>> &connected_components_to_gp_indices_map,
                       Eigen::MatrixXi &quickshift_segments_map,
                       std::vector <std::vector<int>> &quickshift_segments_to_gp_indices_map) {

            const int small_patch_threshold = 10; //0; //msqs_globals::patch_size_threshold; //100;

            const int rows = density_map.rows();
            const int cols = density_map.cols();

            int num_connected_components = connected_components_to_gp_indices_map.size();
            //const auto &point_index_map = ground_hist.point_index_map();

            // This is 2D map of clusters, can be used for visualizations
            quickshift_segments_map.setZero(rows, cols);

            // Clear cluster lists, maps
            quickshift_segments_to_gp_indices_map.clear();

            // Need for visualization?
            Eigen::MatrixXi eqt_map(rows, cols);
            eqt_map.setZero();

            int proposal_id = 1;
            // Loop through connected components, segment them further with Quick-Shift [Vedadi'08]
            for (unsigned i = 0; i < num_connected_components; i++) {
                int cc_idx = i;
                const auto &gp_inds = connected_components_to_gp_indices_map.at(cc_idx);
                const unsigned num_cells_of_component = gp_inds.size();

                // Ignore (small) patches, whose w*h < T
                if (num_cells_of_component < small_patch_threshold)
                    continue;

                // Here, just want to look-up index of the connected component, stored in the CC map. Need just first cell to make the lookup.
                // We need this later, to make sure we do not have cluster duplicates, because some point might end up in two CC crops.
                int cell_u = 0, cell_v = 0;
                SUN::utils::UnravelIndex(gp_inds.at(0), cols, &cell_u, &cell_v);
                int cell_lookup_idx = connected_component_map(cell_v, cell_u);

                // Find width, height of the connected component!
                int max_u = -1;
                int max_v = -1;
                int min_u = 1e5;
                int min_v = 1e5;
                for (const int idx:gp_inds) {
                    int u = 0, v = 0;
                    SUN::utils::UnravelIndex(idx, cols, &u, &v); // col, row
                    if (u > max_u) max_u = u;
                    if (v > max_v) max_v = v;
                    if (u < min_u) min_u = u;
                    if (v < min_v) min_v = v;
                }
                const int cc_width = max_u - min_u;
                const int cc_height = max_v - min_v;
                const int cc_size = cc_width * cc_height;

                // Connected Component region crops
                //const Eigen::MatrixXd &crop = ground_hist.density_map().block(min_v, min_u, cc_height, cc_width);
                const Eigen::MatrixXd &crop = density_map.block(min_v, min_u, cc_height, cc_width);

                int *eqtTable = new int[cc_size]; // Store parent-child relations of the shifts here!
                for (int u = 0; u < crop.cols(); u++) {
                    for (int v = 0; v < crop.rows(); v++) {
                        const int window_min_v = std::max(v - 1, 0);
                        const int window_min_u = std::max(u - 1, 0);
                        const int window_max_v = std::min(v + 1, static_cast<int>(crop.rows() - 1));
                        const int window_max_u = std::min(u + 1, static_cast<int>(crop.cols() - 1));
                        const Eigen::MatrixXd &window_3x3 = crop.block(window_min_v, window_min_u,
                                                                       window_max_v - window_min_v + 1,
                                                                       window_max_u - window_min_u + 1);

                        assert(window_3x3.rows() <= 3);
                        assert(window_3x3.cols() <= 3);

                        Eigen::MatrixXd::Index max_row, max_col; // max_y, max_x
                        window_3x3.maxCoeff(&max_row, &max_col);

                        if (u > 0) max_col -= 1; // Because max_col, max_row should be relative to center, not border
                        if (v > 0) max_row -= 1;

                        // Entry in the equivalence table stores child-parent relations
                        // Index of parent node is stored at index i
                        eqtTable[v * cc_width + u] = (v + max_row) * cc_width + (u + max_col);
                    }
                }

                // Eqt table now stores for each window direction of "best" shift. Now perform "gradient ascent"!
                // I.e. from each cell, walk-up to the mode. As a result, we will store id of the converged-mode (parent cell) for each cell in the table.
                for (int j = 0; j < cc_size; j++) {
                    while (eqtTable[j] != eqtTable[eqtTable[j]]) {
                        eqtTable[j] = eqtTable[eqtTable[j]];
                    }
                }

                // Now transform this cell<->converged-mode table to 2D array
                Eigen::MatrixXi reshapedEqt(cc_height, cc_width);
                for (int u = 0; u < crop.cols(); u++) {
                    for (int v = 0; v < crop.rows(); v++) {
                        reshapedEqt(v, u) = eqtTable[v * cc_width + u];
                        //-------------------------- Keep this?
                        int cell_u = min_u + u;
                        int cell_v = min_v + v;
                        eqt_map(cell_v, cell_u) = eqtTable[v * cc_width + u];
                        //--------------------------
                    }
                }

                // Get a vector of unique labels (modes)!
                std::vector <int> uniqueLabels(eqtTable, eqtTable + cc_size);
                std::sort(uniqueLabels.begin(), uniqueLabels.end());
                uniqueLabels.erase(std::unique(uniqueLabels.begin(), uniqueLabels.end()), uniqueLabels.end());
                delete[] eqtTable;

                // Loop through all clusters. Each will be turned into object proposal.
                for (auto label_id:uniqueLabels) {
                    // Here, we will store relevant indices of the Quick-Shift segment.
                    pcl::PointIndices pointcloud_inds;
                    std::vector <int> gp_inds;

                    // Loop through patch, work-out cells belonging to the seg. obj., fill up the index containers
                    for (int u = 0; u < crop.cols(); u++) {
                        for (int v = 0; v < crop.rows(); v++) {
                            // Fire on each cell that:
                            // - label belongs to labelId
                            // - occ. map > 0
                            // Additional condition:
                            // -make sure we won't process a region, that happens to fall into two CC windows, twice!
                            int current_cell_id = reshapedEqt(v, u);
                            double current_cell_occup = crop/*_unblurred*/(v, u);
                            int current_cell_cc_id = connected_component_map(v + min_v, u +
                                                                                        min_u); //crop_cc_map(v, u); // To which connected component cell belongs?

                            // TODO (Aljosa) cell_occup: > T, rather than 0.0?
                            if ((current_cell_id == label_id) && (current_cell_occup > 0.0) &&
                                (current_cell_cc_id == cell_lookup_idx)) {
                                // This are coords of the cell in the gp hist!
                                int cell_u = min_u + u;
                                int cell_v = min_v + v;
                                quickshift_segments_map(cell_v, cell_u) = proposal_id;

                                // Push gp-index
                                int gp_ind = 0;
                                SUN::utils::RavelIndex(cell_u, cell_v, cols, &gp_ind);
                                gp_inds.push_back(gp_ind);

                                // Push 3d-point-indices
                                const std::vector <int> &pcl_inds_of_the_cell = point_index_map(cell_v, cell_u);
                                pointcloud_inds.indices.insert(pointcloud_inds.indices.end(),
                                                               pcl_inds_of_the_cell.begin(),
                                                               pcl_inds_of_the_cell.end());
                            }
                        }
                    }

                    proposal_id++;

                    //if (pointcloud_inds.indices.size() < object_proposal_globals::g_min_num_points)
                    //    continue;

                    quickshift_segments_to_gp_indices_map.push_back(gp_inds);

                }
            }

            return quickshift_segments_to_gp_indices_map.size();
        }

    }
}

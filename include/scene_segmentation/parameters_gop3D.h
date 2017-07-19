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

#ifndef GOT_PARAMETERS_GOP3D_H
#define GOT_PARAMETERS_GOP3D_H

// boost
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace GOP3D {
    void InitParameters(po::options_description &options) {

        options.add_options()
                // Ground-discretization parameters
                ("ground_histogram_length", po::value<double>()->default_value(40.0), "Ground histogram discretization: length")
                ("ground_histogram_depth", po::value<double>()->default_value(60.0), "Ground histogram discretization: depth")
                ("ground_histogram_height", po::value<double>()->default_value(3.0), "Ground histogram discretization: height")
                ("ground_histogram_rows", po::value<int>()->default_value(600), "Ground histogram discretization: rows")
                ("ground_histogram_cols", po::value<int>()->default_value(400), "Ground histogram discretization: cols")
                ("ground_histogram_noise_threshold", po::value<double>()->default_value(0.2), "Ground histogram nose threshold")

                // Filtering parameters
                ("filtering_min_distance_to_plane", po::value<double>()->default_value(0.1), "Filtering: remove points closer than this.")
                ("filtering_max_distance_to_plane", po::value<double>()->default_value(3.0), "Filtering: remove points farther than this.")
                ("filtering_cluster_density_threshold", po::value<double>()->default_value(0.2), "Filtering: remove clusters that are less dense than this.")
                ("filtering_min_num_points", po::value<int>()->default_value(100), "Filtering: remove clusters that contain less points than specified.")

                // Multi-Scale clustering
                ("clustering_scale_persistence_overlap_threshold", po::value<double>()->default_value(0.8), "Clusters that overlap more than this (across the scales) are merged.")
                ("clustering_num_scale_spaces", po::value<int>()->default_value(10), "Number of scale spaces (per-filter).")
                ("clustering_initial_kernel_radius", po::value<double>()->default_value(2.0), "Initial kernel radius.");
        ;
    }

    void InitLaserParameters(po::options_description &options) {

        options.add_options()
                // Ground-discretization parameters
                ("ground_histogram_length", po::value<double>()->default_value(40.0), "Ground histogram discretization: length")
                ("ground_histogram_depth", po::value<double>()->default_value(60.0), "Ground histogram discretization: depth")
                ("ground_histogram_height", po::value<double>()->default_value(3.0), "Ground histogram discretization: height")
                ("ground_histogram_rows", po::value<int>()->default_value(600), "Ground histogram discretization: rows")
                ("ground_histogram_cols", po::value<int>()->default_value(400), "Ground histogram discretization: cols")
                ("ground_histogram_noise_threshold", po::value<double>()->default_value(0.2), "Ground histogram nose threshold")

                // Filtering parameters
                ("filtering_min_distance_to_plane", po::value<double>()->default_value(0.1), "Filtering: remove points closer than this.")
                ("filtering_max_distance_to_plane", po::value<double>()->default_value(3.0), "Filtering: remove points farther than this.")
                ("filtering_cluster_density_threshold", po::value<double>()->default_value(0.2), "Filtering: remove clusters that are less dense than this.")
                ("filtering_min_num_points", po::value<int>()->default_value(10), "Filtering: remove clusters that contain less points than specified.")

                // Multi-Scale clustering
                ("clustering_scale_persistence_overlap_threshold", po::value<double>()->default_value(0.8), "Clusters that overlap more than this (across the scales) are merged.")
                ("clustering_num_scale_spaces", po::value<int>()->default_value(10), "Number of scale spaces (per-filter).")
                ("clustering_initial_kernel_radius", po::value<double>()->default_value(0.2), "Initial kernel radius.");
        ;
    }
}

#endif //GOT_PARAMETERS_GOP3D_H

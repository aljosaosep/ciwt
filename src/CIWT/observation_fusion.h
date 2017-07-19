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

#ifndef GOT_OBSERVATIONS_PROCESSING
#define GOT_OBSERVATIONS_PROCESSING

// eigen
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// boost
#include <boost/program_options.hpp>

// opencv
#include <opencv2/core/core.hpp>

// tracking lib
#include <tracking/observation.h>
#include <tracking/utils_tracking.h>

// segmentation
#include <scene_segmentation/object_proposal.h>

// Forward decl.
namespace SUN { class Calibration; }
namespace SUN { namespace utils { class Camera; } }
namespace SUN { namespace utils { class Detection; } }

namespace po = boost::program_options;

namespace GOT {
    namespace tracking {
        namespace observation_processing {

            namespace category_size_stats {
                // Obtained using MLE on the train split of the KITTI training dataset
                const double car_mean_whl[] = {1.659722, 1.595054, 4.031461};
                const double car_variance_whl[] = {0.138496, 0.282258, 0.583113};

                const double ped_mean_whl[] = {0.667334, 1.791161, 0.902490};
                const double ped_variance_whl[] = {0.144467, 0.115740, 0.238094};

                const double cyclist_mean_whl[] = {0.632071, 1.717793, 1.706747};
                const double cyclist_variance_whl[] = {0.169315, 0.063247, 0.120272};
            }

            const double proposals_variance_size_estimates[] = {0.1, 0.2, 1.0};

            struct AssocContext {
                double score;
                int det_idx;
                int proposal_idx;

                AssocContext() {
                    score = 0.0;
                    det_idx = -1;
                    proposal_idx = -1;
                }
            };

            std::vector<GOT::tracking::Observation> DetectionToSegmentationFusion(
                    const std::vector<SUN::utils::Detection> &detections_in,
                    const std::vector<GOT::segmentation::ObjectProposal> &proposals_in,
                    const SUN::utils::Camera &camera,
                    const cv::Mat &image,
                    const po::variables_map &parameter_map);

            std::vector<GOT::tracking::Observation> DetectionsOnly (
                    const std::vector <SUN::utils::Detection> &detections_in,
                    const std::vector<GOT::segmentation::ObjectProposal> &proposals_in,
                    const cv::Mat &image);
        }
    }
}

#endif
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

#ifndef GOT_TRACKING_VISUALIZATION
#define GOT_TRACKING_VISUALIZATION

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

// tracking
#include <tracking/hypothesis.h>

// utils
#include "sun_utils/utils_kitti.h"

// fwd. decl.
namespace SUN { namespace utils { class Detection; }}
namespace SUN { namespace utils { class Camera; }}

namespace GOT {
    namespace tracking {
        typedef std::function<void(const GOT::tracking::Hypothesis&, const SUN::utils::Camera&, cv::Mat&)> DrawHypoFnc;
        typedef std::function<void(const GOT::tracking::Observation&, const SUN::utils::Camera&, cv::Mat&, int)> DrawObsFnc;

        namespace draw_hypos {
            void DrawTrajectoryToGroundPlane(const std::vector<Eigen::Vector4d> &poses,  const SUN::utils::Camera &camera, const cv::Scalar &color, cv::Mat &ref_image, int line_width=1, int num_poses_to_draw=20, int smoothing_window_size=20);
            void DrawHypothesis2d(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera, cv::Mat &ref_image);
            void DrawHypothesis3d(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera, cv::Mat &ref_image);
        }

        namespace draw_observations {
            void DrawObservationAndOrientation(const GOT::tracking::Observation &observation, const SUN::utils::Camera &cam, cv::Mat &ref_img, int index);
            void DrawObservationPaper(const GOT::tracking::Observation &observation, const SUN::utils::Camera &cam, cv::Mat &ref_img, int index);
            void DrawObservationByID(const GOT::tracking::Observation &observation, const SUN::utils::Camera &cam, cv::Mat &ref_img, int index);
        }

        class Visualizer {

        public:
            Visualizer();

            struct VisualizationProperties {
                // Bird-eye visualization
                float birdeye_scale_factor_;
                int birdeye_far_plane_;
                int birdeye_left_plane_;
                int birdeye_right_plane_;

                VisualizationProperties() {
                    // Bird-eye visualization module params
                    birdeye_scale_factor_ = 13.0;
                    birdeye_far_plane_ = 80;
                    birdeye_left_plane_ = -30;
                    birdeye_right_plane_ = 30;
                }
            };

            const void GetColor(int index, double &r, double &g, double &b) const;

            // -------------------------------------------------------------------------------
            // +++ 3D VISUALIZER METHODS +++
            // -------------------------------------------------------------------------------

            /**
              * @brief Renders 3D bounding-box representation of the hypothesis.
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            void RenderHypo3D(pcl::visualization::PCLVisualizer &viewer, const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,  const int viewport);


            /**
              * @brief Renders trajectory of the tracked object.
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            void RenderTrajectory(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,  const std::string &traj_id, double r, double g, double b, pcl::visualization::PCLVisualizer &viewer, int viewport=0);

            // -------------------------------------------------------------------------------
            // +++ DRAW HYPOS/OBSERVATIONS +++
            // -------------------------------------------------------------------------------
            void DrawHypotheses(const std::vector<GOT::tracking::Hypothesis> &hypos, const SUN::utils::Camera &camera, cv::Mat &ref_image, DrawHypoFnc draw_hypo_fnc) const;
            void DrawObservations(const std::vector<GOT::tracking::Observation> &observations, cv::Mat &ref_img, const SUN::utils::Camera &cam, DrawObsFnc draw_obs_fnc) const;

            // -------------------------------------------------------------------------------
            // +++ TOP-DOWN VISUALIZATION TOOLS +++
            // -------------------------------------------------------------------------------

            /**
              * @brief Draws top-down projection of the point-cloud.
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            void DrawPointCloudProjectionGroundPlane(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud, const SUN::utils::Camera &camera, cv::Mat &ref_image) const;

            /**
              * @brief Draws grid to the ground.
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            void DrawGridBirdeye(double res_x, double res_z, const SUN::utils::Camera &camera, cv::Mat &ref_image) const;

            /**
              * @brief Transf. 2D ground-plane point into the 'scaled frustum'
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            void TransformPointToScaledFrustum(double &pose_x, double &pose_z) const;

            /**
              * @brief Top-down view of observations.
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            void DrawObservationsBirdEye(const std::vector<GOT::tracking::Observation> &observations, pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                         const SUN::utils::Camera &camera, cv::Mat &ref_image) const;

            /**
              * @brief Top-down view of trajectories.
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            void DrawTrajectoriesBirdEye(const GOT::tracking::HypothesesVector &hypos,
                                         pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                         const SUN::utils::Camera &camera, cv::Mat &ref_image) const;

            // -------------------------------------------------------------------------------
            // +++ TOOLS +++
            // -------------------------------------------------------------------------------

            /**
             * @brief Computes vertices of oriented 3D bounding box, that represents the tracked object.
             * @author Aljosa (osep@vision.rwth-aachen.de)
             */
            static std::vector<Eigen::Vector3d> ComputeOrientedBoundingBoxVertices(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera);


            // Etc
            void set_visualization_propeties(const VisualizationProperties &visualization_props);

        private:
            VisualizationProperties visualization_propeties_;
        };

    }
}

#endif

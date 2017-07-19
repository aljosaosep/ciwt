/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Dirk Klostermann (osep, klostermann -at- vision.rwth-aachen.de)

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

#ifndef SUN_UTILS_VISUALIZATION
#define SUN_UTILS_VISUALIZATION

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

// project
#include "camera.h"

// Forward declarations
namespace SUN { namespace utils { class Detection; } }

namespace SUN {
    namespace utils {
        namespace visualization {
            // -------------------------------------------------------------------------------
            // +++ COLOR TABLES +++
            // -------------------------------------------------------------------------------
            void GenerateColor(unsigned int id, uint8_t &r, uint8_t &g, uint8_t &b);
            void GenerateColor(unsigned int id, cv::Vec3f &color);
            void GenerateColor(unsigned int id, cv::Vec3b &color);

            // -------------------------------------------------------------------------------
            // +++ PRIMITIVES +++
            // -------------------------------------------------------------------------------
            void DrawLine(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const SUN::utils::Camera &camera, cv::Mat &ref_image, const cv::Scalar &color, int thickness=1, int line_type=1, const cv::Point2i &offset=cv::Point2i(0,0));
            void DrawBoundingBox2d(const Eigen::VectorXd &bounding_box_2d, cv::Mat &ref_image, uint8_t r=255, uint8_t g=0, uint8_t b=0, int thickness=1.0);
            void DrawObjectFilled(const std::vector<int> &indices, const Eigen::Vector4d &bounding_box_2d, const cv::Vec3b &color, double alpha, cv::Mat &ref_image);
            void ArrowedLine(cv::Point2d pt1, cv::Point2d pt2, const cv::Scalar& color, cv::Mat &ref_image, int thickness=1, int line_type=8, int shift=0, double tipLength=0.1);

            // -------------------------------------------------------------------------------
            // +++ COVARIANCE MATRICES +++
            // -------------------------------------------------------------------------------
            /**
              * @brief Draws an iso-contour of the covariance matrix (iso-contour is picked via chisquare_val)
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            void DrawCovarianceMatrix2dEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat, cv::Mat &ref_image, cv::Vec3f color);

            // -------------------------------------------------------------------------------
            // +++ ETC +++
            // -------------------------------------------------------------------------------

            /**
               * @brief Draw object detections to the image.
               * @author Aljosa (osep@vision.rwth-aachen.de)
               */
            void DrawDetections(const std::vector<SUN::utils::Detection> &detections, cv::Mat &ref_image, int offset=0);

            void RenderBoundingBox3D(pcl::visualization::PCLVisualizer &viewer, const Eigen::VectorXd &bbox, double r, double g, double b, std::string &id, const int viewport=0);

            /// Bird-eye visualization tools
            void TransformPointToScaledFrustum(double &pose_x, double &pose_z, const double left_plane, const double scale_factor);
        }
    }
}

#endif

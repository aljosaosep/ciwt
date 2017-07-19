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


#include "utils_visualization.h"

// std
#include <random>

// OpenCV
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>

// pcl
#include <pcl/common/transforms.h>

// utils
#include "detection.h"

// eigen
#include <Eigen/Dense>

// Project
#include "camera.h"
#include "sun_utils/utils_visualization.h"
#include "utils_common.h"
#include "ground_model.h"

// Boost
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

namespace Eigen {
    namespace internal {
        template<typename Scalar>
        struct scalar_normal_dist_op
        {
            static boost::mt19937 rng;    // The uniform pseudo-random algorithm
            mutable boost::normal_distribution<Scalar> norm;  // The gaussian combinator

            EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

            template<typename Index>
            inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
        };

        template<typename Scalar> boost::mt19937 scalar_normal_dist_op<Scalar>::rng;

        template<typename Scalar>
        struct functor_traits<scalar_normal_dist_op<Scalar> >
        { enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };
    } // end namespace internal
} // end namespace Eigen

namespace SUN {
    namespace utils {
        namespace visualization {

            const int color_array_size = 50;
            unsigned char color_array[] = {
                    240, 62, 36,
                    245, 116, 32,
                    251, 174, 24,
                    213, 223, 38,
                    153, 204, 112,
                    136, 201, 141,
                    124, 201, 169,
                    100, 199, 230,
                    64, 120, 188,
                    61, 88, 167,
                    204,     0,   255,
                    255,     0,     0,
                    0,   178,   255,
                    255,     0,   191,
                    255,   229,     0,
                    0,   255,   102,
                    89,   255,     0,
                    128,     0,   255,
                    242,     0,   255,
                    242,   255,     0,
                    255,     0,    77,
                    51,     0,   255,
                    0,   255,   140,
                    0,   255,    25,
                    204,   255,     0,
                    255,   191,     0,
                    89,     0,   255,
                    0,   217,   255,
                    0,    64,   255,
                    255,   115,     0,
                    255,     0,   115,
                    166,     0,   255,
                    13,     0,   255,
                    0,    25,   255,
                    0,   255,   217,
                    0,   255,    64,
                    255,    38,     0,
                    255,     0,   153,
                    0,   140,   255,
                    255,    77,     0,
                    255,   153,     0,
                    0,   255,   179,
                    0,   102,   255,
                    255,     0,    38,
                    13,   255,     0,
                    166,   255,     0,
                    0,   255,   255,
                    128,   255,     0,
                    255,     0,   230,
                    51,   255,     0
            };

            void GenerateColor(unsigned int id,  uint8_t &r, uint8_t &g, uint8_t &b) {
                int col_index = id;
                if (col_index > color_array_size)
                    col_index = col_index%color_array_size;
                int seed_idx = col_index*3;
                r = static_cast<uint8_t>(color_array[seed_idx]);
                g = static_cast<uint8_t>(color_array[seed_idx+1]);
                b = static_cast<uint8_t>(color_array[seed_idx+2]);
            }

            void GenerateColor(unsigned int id, cv::Vec3f &color) {
                uint8_t r,g,b;
                GenerateColor(id, r,g,b);
                color = cv::Vec3f(static_cast<float>(b)/255.0f, static_cast<float>(g)/255.0f, static_cast<float>(r)/255.0f);
            }

            void GenerateColor(unsigned int id, cv::Vec3b &color) {
                uint8_t r,g,b;
                GenerateColor(id, r,g,b);
                color = cv::Vec3b(b, g, r);
            }

            void DrawObjectFilled(const std::vector<int> &indices, const Eigen::Vector4d &bounding_box_2d, const cv::Vec3b &color, double alpha, cv::Mat &ref_image) {
                SUN::utils::visualization::DrawBoundingBox2d(bounding_box_2d, ref_image, color[2], color[1], color[0]);

                // Draw overlay over proposal region
                for (auto ind:indices) {
                    int x, y;
                    SUN::utils::UnravelIndex(ind, ref_image.cols, &x, &y);
                    if (y > 0 && x > 0 && y < ref_image.rows &&
                        x < ref_image.cols)
                        ref_image.at<cv::Vec3b>(y, x) =
                                ref_image.at<cv::Vec3b>(y, x) * alpha +
                                (1 - alpha) * color;
                }
            }

            void ArrowedLine(cv::Point2d pt1, cv::Point2d pt2, const cv::Scalar& color, cv::Mat &ref_image, int thickness, int line_type, int shift, double tipLength) {
                const double tipSize = cv::norm(pt1-pt2)*tipLength; // Factor to normalize the size of the tip depending on the length of the arrow
                cv::line(ref_image, pt1, pt2, color, thickness, line_type, shift);
                const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );
                cv::Point2d p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)), cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
                cv::line(ref_image, p, pt2, color, thickness, line_type, shift);
                p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
                p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
                cv::line(ref_image, p, pt2, color, thickness, line_type, shift);
            }

            void DrawLine(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const SUN::utils::Camera &camera, cv::Mat &ref_image, const cv::Scalar &color, int thickness, int line_type, const cv::Point2i &offset) {
                Eigen::Vector4d p1_4d, p2_4d;
                p1_4d[3] = p2_4d[3] = 1.0;
                p1_4d.head<3>() = p1;
                p2_4d.head<3>() = p2;
                Eigen::Vector3i projected_point_1 = camera.CameraToImage(p1_4d);
                Eigen::Vector3i projected_point_2 = camera.CameraToImage(p2_4d);
                auto cv_p1 = cv::Point2i(projected_point_1[0],projected_point_1[1]);
                auto cv_p2 = cv::Point2i(projected_point_2[0],projected_point_2[1]);

                bool p1_in_bounds = true;
                bool p2_in_bounds = true;
                if ((cv_p1.x < 0) && (cv_p1.y < 0) && (cv_p1.x > ref_image.cols) && (cv_p1.y > ref_image.rows) )
                    p1_in_bounds = false;

                if ((cv_p2.x < 0) && (cv_p2.y < 0) && (cv_p2.x > ref_image.cols) && (cv_p2.y > ref_image.rows) )
                    p2_in_bounds = false;

                // Draw line, but only if both end-points project into the image!
                if (p1_in_bounds || p2_in_bounds) { // This is correct. Won't draw only if both lines are out of bounds.
                    // Draw line
                    auto p1_offs = offset+cv_p1;
                    auto p2_offs = offset+cv_p2;
                    if (cv::clipLine(cv::Size(/*0, 0, */ref_image.cols, ref_image.rows), p1_offs, p2_offs)) {
                        cv::line(ref_image, p1_offs, p2_offs, color, thickness, line_type);
                    }
                }
            }

            void DrawBoundingBox2d(const Eigen::VectorXd &bounding_box_2d, cv::Mat &ref_image, uint8_t r, uint8_t g, uint8_t b, int thickness) {
                cv::Rect rect(cv::Point2d(bounding_box_2d[0],bounding_box_2d[1]), cv::Size(bounding_box_2d[2], bounding_box_2d[3]));
                cv::rectangle(ref_image, rect, cv::Scalar(b, g, r), thickness);
            }

            void DrawBoundingBox3d(const Eigen::VectorXd &bounding_box_3d, cv::Mat &ref_image, const SUN::utils::Camera &camera, uint8_t r, uint8_t g, uint8_t b) {
                // Params
                const int line_width = 2;
                auto cv_color_triplet = cv::Scalar(r, g, b);

                // Center
                Eigen::Vector4d center;
                center.head(3) = bounding_box_3d.head(3);
                center[3] = 1.0;

                // Direction vector, obtain from the hypo points.
                Eigen::Vector3d dir(0.0, 0.0, 1.0); // Fixed, frontal dir. //= //hypo.GetDirectionVector(4);

                // Dimensions
                auto width = bounding_box_3d[3];
                auto height = bounding_box_3d[4];
                auto length = bounding_box_3d[5];

                // Bounding box is defined by up-vector, direction vector and a vector, orthogonal to the two.
                Eigen::Vector3d ground_plane_normal(0.0,-1.0,0.0);
                Eigen::Vector3d ort_to_dir_vector = dir.cross(ground_plane_normal);
                Eigen::Vector3d center_proj_to_ground_plane = center.head(3);
                center_proj_to_ground_plane = camera.ground_model()->ProjectPointToGround(center_proj_to_ground_plane);

                // Re-compute dir and or vectors
                Eigen::Vector3d dir_recomputed = ort_to_dir_vector.cross(ground_plane_normal); //ort_to_dir_vector.cross(ground_plane_normal);
                Eigen::Vector3d ort_recomputed = ground_plane_normal.cross(dir_recomputed);

                // Scale these vectors by bounding-box dimensions
                Eigen::Vector3d dir_scaled = dir_recomputed.normalized() * (length/2.0) * -1.0;
                Eigen::Vector3d ort_scaled = ort_recomputed.normalized() * (width/2.0);
                Eigen::Vector3d ground_plane_normal_scaled = ground_plane_normal*height;

                std::vector<Eigen::Vector3d> rect_3d_points(8);
                std::vector<int> rect_3d_visibility_of_points(8);

                // Render 3d bounding-rectangle into the image
                // Compute 8 corner of a rectangle
                rect_3d_points.at(0) = center_proj_to_ground_plane + dir_scaled + ort_scaled;
                rect_3d_points.at(1) = center_proj_to_ground_plane - dir_scaled + ort_scaled;
                rect_3d_points.at(2) = center_proj_to_ground_plane + dir_scaled - ort_scaled;
                rect_3d_points.at(3) = center_proj_to_ground_plane - dir_scaled - ort_scaled;
                rect_3d_points.at(4) = center_proj_to_ground_plane + dir_scaled + ort_scaled + ground_plane_normal_scaled;
                rect_3d_points.at(5) = center_proj_to_ground_plane - dir_scaled + ort_scaled + ground_plane_normal_scaled;
                rect_3d_points.at(6) = center_proj_to_ground_plane + dir_scaled - ort_scaled + ground_plane_normal_scaled;
                rect_3d_points.at(7) = center_proj_to_ground_plane - dir_scaled - ort_scaled + ground_plane_normal_scaled;

                // Compute point visibility
                for (int i=0; i<8; i++) {
                    Eigen::Vector4d pt_4d;
                    pt_4d.head<3>() = rect_3d_points.at(i);
                    pt_4d[3] = 1.0;
                    rect_3d_visibility_of_points.at(i) = camera.IsPointInFrontOfCamera(pt_4d);
                }

                // Render lines
                DrawLine(rect_3d_points.at(0),rect_3d_points.at(1), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(1),rect_3d_points.at(3), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(3),rect_3d_points.at(2), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(2),rect_3d_points.at(0), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(4),rect_3d_points.at(5), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(5),rect_3d_points.at(7), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(7),rect_3d_points.at(6), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(6),rect_3d_points.at(4), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(0),rect_3d_points.at(4), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(1),rect_3d_points.at(5), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(2),rect_3d_points.at(6), camera, ref_image, cv_color_triplet, line_width, 1);
                DrawLine(rect_3d_points.at(3),rect_3d_points.at(7), camera, ref_image, cv_color_triplet, line_width, 1);
            }

            void RenderBoundingBox3D(pcl::visualization::PCLVisualizer &viewer, const Eigen::VectorXd &bbox, double r, double g, double b, std::string &id, const int viewport) {

                assert(bbox.size()>=6);

                const double w_by_2 = bbox[3]/2.0;
                const double h_by_2 = bbox[4]/2.0;
                const double l_by_2 = bbox[5]/2.0;
                const double cx = bbox[0];
                const double cy = bbox[1];
                const double cz = bbox[2];

                std::vector<pcl::PointXYZ> pts3d(8);
                pts3d[0] = pcl::PointXYZ(cx+w_by_2, cy-h_by_2,cz-l_by_2); // 1
                pts3d[1] = pcl::PointXYZ(cx+w_by_2, cy+h_by_2,cz-l_by_2); // 2
                pts3d[2] = pcl::PointXYZ(cx-w_by_2, cy+h_by_2,cz-l_by_2); // 3
                pts3d[3] = pcl::PointXYZ(cx-w_by_2, cy-h_by_2,cz-l_by_2); // 4
                pts3d[4] = pcl::PointXYZ(cx+w_by_2, cy-h_by_2,cz+l_by_2); // 5
                pts3d[5] = pcl::PointXYZ(cx+w_by_2, cy+h_by_2,cz+l_by_2); // 6
                pts3d[6] = pcl::PointXYZ(cx-w_by_2, cy+h_by_2,cz+l_by_2); // 7
                pts3d[7] = pcl::PointXYZ(cx-w_by_2, cy-h_by_2,cz+l_by_2); // 8

                Eigen::Vector3f center(cx,cy,cz);

                Eigen::Quaternionf q;
                if (bbox.size() == 10) {
                    q = Eigen::Quaternionf(bbox[6], bbox[7], bbox[8], bbox[9]);
                }
                else {
                    q = Eigen::Quaternionf::Identity();
                }
                viewer.addCube(center, q, bbox[3], bbox[4], bbox[5], id, viewport);

                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3.0, id, viewport);
                viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, r,g,b, id);
                viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 1.0, id);
                viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
            }


            void DrawDetections(const std::vector<SUN::utils::Detection> &detections, cv::Mat &ref_image, int offset) {
                cv::Point2i offs2(offset, offset);
                //cv::copyMakeBorder(ref_image, ref_image, offs2.x, offs2.y, offs2.x, offs2.y, cv::BORDER_CONSTANT, cv::Scalar(255,255,255));

                // For observation score!
                const double minimum = 0.0;
                const double maximum = 1.0;

                auto heatmap_func = [](double value, double minimum, double maximum, uint8_t &r, uint8_t &g, uint8_t &b) {
                    double ratio = 2 * (value-minimum) / (maximum - minimum);
                    b = static_cast<uint8_t>(std::max(0.0, 255*(1.0 - ratio)));
                    r = static_cast<uint8_t>(std::max(0.0, 255*(ratio - 1.0)));
                    g = static_cast<uint8_t>(255 - b - r);
                };

                //DrawHeatmapBar(ref_image, minimum, maximum, heatmap_func);

                int det_index = 0;
                for (const auto &det:detections) {
                    Eigen::Vector4d bb_2d = det.bounding_box_2d();
                    cv::Rect rect(cv::Point2i(bb_2d[0],bb_2d[1])+offs2, cv::Size(bb_2d[2], bb_2d[3]));

                    // ------------- GEN COLOR ~ COLOR_CODE SCORE ------------
                    uint8_t r,g,b;
                    heatmap_func(det.score(), minimum, maximum, r,g,b);
                    // ------------------------------------------------------

                    cv::rectangle(ref_image, rect, cv::Scalar(b,g,r), 2);

                    int category = det.category(); // Fifth entry is category!

                    std::string category_str = "other";
                    if (category==0)
                        category_str = "car";
                    else if (category==1)
                        category_str = "van";
                    else if (category==2)
                        category_str = "truck";
                    else if (category==3)
                        category_str = "ped.";
                    else if (category==5)
                        category_str = "cyc.";

                    cv::putText(ref_image,  category_str, cv::Point2d(offs2.x+bb_2d[0]+5, offs2.y+bb_2d[1]+20), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(0,0,255), 2);

                    //char buff[50];
                    //snprintf(buff, 50, "%1.2f", det.score());
                    //cv::putText(ref_image,  buff, cv::Point2d(offset.x+bb_2d[0]+5, offset.y+bb_2d[1]+40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,0,255), 1);
                    cv::putText(ref_image,  std::to_string(det_index), cv::Point2d(offset+bb_2d[0]+5, offset+bb_2d[1]+60), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,0,255), 1);
                    det_index ++;
                }
            }

            void TransformPointToScaledFrustum(double &pose_x, double &pose_z, const double left_plane, const double scale_factor) {
                pose_x += (-left_plane);
                pose_x *= scale_factor;
                pose_z *= scale_factor;
            }



            // Adapted from Vincent Spruyt
            void DrawCovarianceMatrix2dEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat, cv::Mat &ref_image, cv::Vec3f color) {
                // Get the eigenvalues and eigenvectors
                cv::Mat eigenvalues, eigenvectors;
                cv::eigen(covmat, eigenvalues, eigenvectors);

                //Calculate the angle between the largest eigenvector and the x-axis
                double angle = atan2(eigenvectors.at<double>(0,1), eigenvectors.at<double>(0,0));

                // Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
                if(angle < 0)
                    angle += 6.28318530718;

                // Convert to degrees instead of radians
                angle = 180*angle/3.14159265359;

                // Calculate the size of the minor and major axes
                double half_majoraxis_size=chisquare_val*sqrt(std::abs(eigenvalues.at<double>(0)));
                double half_minoraxis_size=chisquare_val*sqrt(std::abs(eigenvalues.at<double>(1)));

                // Return the oriented ellipse
                // The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
                cv::RotatedRect rot_rect(mean, cv::Size2f(half_majoraxis_size, half_minoraxis_size), /*-*/angle);
                cv::ellipse(ref_image,  rot_rect, cv::Scalar(color[0], color[1], color[2]), 1);
            }

        }
    }
}

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

#ifndef GOT_DETECTION_H
#define GOT_DETECTION_H

// eigen
#include <Eigen/Core>

// boost
#include <boost/serialization/access.hpp>

// std
#include <memory>
#include <set>

// boost
#include <boost/program_options.hpp>

// Forward decl.
namespace SUN { namespace utils { class Camera; }};
namespace SUN { namespace utils { namespace KITTI { class Calibration; }}};
namespace cv { class Mat; };

namespace po = boost::program_options;

namespace SUN {
    namespace utils {

        class Detection {
//            //! For serialization.
//            friend class boost::serialization::access;
//
//            template<class Archive>
//            void serialize(Archive &ar, const unsigned int version) {
//                ar & category_;
//                ar & id_;
//                ar & score_;
//                ar & footpoint_;
//                ar & bounding_box_2d_;
//                ar & bounding_box_3d_;
//                ar & pose_covariance_matrix_;
//            }

        public:
            Detection();

            Detection(const Eigen::Vector4d &bounding_box_2d, int detection_id);

            // Getters
            const Eigen::Vector4d &bounding_box_2d() const;
            const Eigen::VectorXd &bounding_box_3d() const;
            const Eigen::Vector4d &footpoint() const;
            const Eigen::Matrix3d &pose_covariance_matrix() const;
            int category() const;
            double score() const;
            double observation_angle() const;

            // Setters
            void set_footpoint(const Eigen::Vector4d &footpoint);
            void set_score(double score);
            void set_pose_covariance_matrix(const Eigen::Matrix3d &pose_covariance_matrix);
            void set_category(int detection_category);
            void set_bounding_box_2d(const Eigen::Vector4d &bounding_box_2d);
            void set_bounding_box_3d(const Eigen::VectorXd &bounding_box_3d);
            void set_observation_angle(double observation_angle);

        private:
            Eigen::Matrix3d pose_covariance_matrix_;
            Eigen::Vector4d bounding_box_2d_;
            Eigen::VectorXd bounding_box_3d_;
            Eigen::Vector4d footpoint_;
            double observation_angle_; // This is obs. angle, not orientation!
            int category_;
            int id_;
            double score_;
        };

        namespace detection {

            /**
               * @brief Filter-out detections based in their scores
               * @author Author: Aljosa (osep@vision.rwth-aachen.de).
               */
            std::vector<Detection> ScoreFilter(const std::vector<Detection> &detections_in, std::function<bool(const Detection &detection)> f_score_filt);

            /**
               * @brief Filter-out detections based in their (estimated) 3D size
               * @author Author: Aljosa (osep@vision.rwth-aachen.de).
               */
            std::vector<Detection> GeometricFilter(const std::vector<Detection> &detections_in, const SUN::utils::Camera &camera,
                                                   double apparent_size_max=3.0,
                                                   double apparent_size_min=0.8,
                                                   double max_width_percentage=0.5);

            /**
               * @brief Greedily apply non-maxima suppression on a set of detections (based on rectange-IOU crit.).
               * @params detectons_in Set if input-detections.
               * @params iou_threshold Intersection-over-union thresh, typically in range [0.3, 0.6].
               * @author Author: Aljosa (osep@vision.rwth-aachen.de).
               */
            std::vector<Detection> NonMaximaSuppression(const std::vector<Detection> &detections_in, double iou_threshold=0.6);

            /**
               * @brief Coarsly localize detection in 3D based on ground estimate
               * @author Author: Aljosa (osep@vision.rwth-aachen.de).
               */
            std::vector<Detection> ProjectTo3D(const std::vector<Detection> &detections_in,
                                               const SUN::utils::Camera &left_camera,
                                               const SUN::utils::Camera &right_camera);
        }
    }
}



#endif //GOT_DETECTION_H

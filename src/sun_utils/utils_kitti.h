/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Francis Engelmann, Aljosa Osep  (engelmann, osep -at- vision.rwth-aachen.de)

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

#ifndef GOT_UTILS_KITTI_H
#define GOT_UTILS_KITTI_H

// Eigen includes
#include <Eigen/Dense>

// utils
#include "shared_types.h"

// C/C++ includes
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>

namespace SUN {
    namespace utils {
        namespace KITTI {

            typedef enum LabelOcclusion
            {
                NOT_SPECIFIED = -1,
                FULLY_VISIBLE = 0,
                PARTLY_OCCLUDED = 1,
                LARGELY_OCCLUDED = 2,
                UNKNOWN = 3
            } LabelOccluded;

            //! Structs
            struct TrackingLabel
            {
                unsigned int frame;       // The frame in which this label occurs
                int trackId;              // The tracklet id, same physical objects have same trackId, can be -1 if dont care
                SUN::shared_types::CategoryTypeKITTI type;
                float truncated;
                LabelOccluded occluded;
                double alpha;             // Not used here
                double boundingBox2D[4];  // Pixel coordinates of 2d bounding box - not used in this code
                double dimensions[3];     // Height, width, depth [m] in 3D
                double location[3];       // XYZ position [m] in 3D
                double rotationY;         // Rotation around y-axe
                double score;
            };


            // -------------------------------------------------------------------------------
            // +++ LabelsIO Class +++
            // -------------------------------------------------------------------------------

            /**
              * @brief Class used for parsing KITTI dataset calibration files.
              * @author Francis Engelmann (englemann@vision.rwth-aachen.de)
              */
            class LabelsIO
            {
            public:
                /**
                 * @brief empty IO constructor
                 */
                LabelsIO();

                /**
                 * @brief readLabels
                 * @param filename
                 * @param labels
                 * @return the number of distinctive objects (i.e. tracks) in the scene
                 */
                static int ReadLabels(const std::string &filename, std::vector<TrackingLabel> &labels);

                // This parses 3DOP detections in Francis/Joerg format. Duplicate func, so remove when not needed anymore!
                //static int ReadLabels3DOP(const std::string &filename, std::vector<TrackingLabel> &labels);

                /**
                 * @brief ReadLabels
                 * @param labels
                 * @param filename
                 * @return the number of distinctive objects (i.e. tracks) in the scene
                 */
                static int WriteLabels(std::vector<TrackingLabel> &labels, const std::string &filename);

                /**
                 * @brief ConvertLineToMatrix
                 * @param line[in] the space-seperated line
                 * @param matrix[out] the output 3x4 matrix
                 */
                static void ConvertLineToMatrix(std::string &line, Eigen::Matrix<double, 3, 4> &matrix);
                static void ConvertLineToMatrix(std::string &line, Eigen::Matrix<double, 4, 4> &matrix);

            protected:
                void WriteStringToFile(std::string text, std::ofstream &file);
            };

            // -------------------------------------------------------------------------------
            // +++ Calibration Class +++
            // -------------------------------------------------------------------------------

            /**
              * @brief Class used for parsing KITTI dataset calibration files.
              * @author Francis Engelmann (englemann@vision.rwth-aachen.de)
              */
            class Calibration
            {
            public:
                /**
                 * @brief empty constructor
                 */
                Calibration();

                Calibration(const Calibration &calibration);

                /**
                 * @brief Calibration constructor, reads specified calib file
                 * @param filename - the path e.g. training/calib/0000.txt
                 */
                explicit Calibration(const std::string& filename);

                /**
                 * @brief Opens and reads the specified calib file
                 * @param filename - the path e.g. training/calib/0000.txt
                 * @return wether speciifed file could be opened or not
                 */
                bool Open(const std::string &filename);

                /**
                 * @brief reads the focal-length from
                 * @return focal-length
                 */
                double f(void) const { return P_(0,0); }

                /**
                 * @brief c_u
                 * @return Principal-point (horizontal direction)
                 */
                double c_u(void) const { return P_(0,2); }

                /**
                 * @brief c_v
                 * @return Principal-point (vertical direction)
                 */
                double c_v(void) const { return P_(1,2); }

                /**
                 * @brief computes the base-line from pose matrix
                 * @return base-line between color cameras
                 */
                double b(void) const;

                Eigen::Matrix<double, 4, 4> GetTr_cam0_cam2() const;

                Eigen::Matrix<double,3,3> GetK(void) const { return proj_cam_2_.block<3,3>(0,0);}
                Eigen::Matrix<double,3,4> GetP(void) const { return P_; }
                Eigen::Matrix<double,3,4> GetProjCam0(void) const { return proj_cam_0_; }
                Eigen::Matrix<double,3,4> GetProjCam1(void) const { return proj_cam_1_; }
                Eigen::Matrix<double,3,4> GetProjCam2(void) const { return proj_cam_2_; }
                Eigen::Matrix<double,3,4> GetProjCam3(void) const { return proj_cam_3_; }

                Eigen::Matrix<double,3,1> GetPosCam0(void) const { return pos_cam_0_; }
                Eigen::Matrix<double,3,1> GetPosCam1(void) const { return pos_cam_1_; }
                Eigen::Matrix<double,3,1> GetPosCam2(void) const { return pos_cam_2_; }
                Eigen::Matrix<double,3,1> GetPosCam3(void) const { return pos_cam_3_; }

                Eigen::Matrix<double,4,4> getR_rect(void) const { return R_rect_; }
                Eigen::Matrix<double,4,4> getTr_velo_cam(void) const { return Tr_velo_cam_; }
                Eigen::Matrix<double,4,4> getTr_imu_velo(void) const { return Tr_imu_velo_; }

                void computeCameraCenters();

            private:
                std::ifstream calib_file;

                // Projection matrices, 3x4
                Eigen::Matrix<double,3,4> P_;
                Eigen::Matrix<double,3,4> proj_cam_0_;
                Eigen::Matrix<double,3,4> proj_cam_1_;
                Eigen::Matrix<double,3,4> proj_cam_2_;
                Eigen::Matrix<double,3,4> proj_cam_3_;

                // [R|t] matrices, 4x4
                Eigen::Matrix<double,4,4> R_rect_;
                Eigen::Matrix<double,4,4> Tr_velo_cam_;
                Eigen::Matrix<double,4,4> Tr_imu_velo_;

                // Position of camera centers
                Eigen::Matrix<double,3,1> pos_cam_0_;
                Eigen::Matrix<double,3,1> pos_cam_1_;
                Eigen::Matrix<double,3,1> pos_cam_2_;
                Eigen::Matrix<double,3,1> pos_cam_3_;

                double baseline_;
            };

        }
    }
}


#endif //GOT_UTILS_KITTI_H

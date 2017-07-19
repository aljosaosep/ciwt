/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Francis Engelmann (osep, engelmann -at- vision.rwth-aachen.de)

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

#ifndef SUN_UTILS_IO_H
#define SUN_UTILS_IO_H

// C/C++ includes
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>

// Boost
#include <boost/program_options.hpp>

// Eigen
#include <Eigen/Core>

// OpenCV
#include <opencv2/core/core.hpp>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>

// Forward declarations
namespace SUN { class DisparityMap; }
namespace SUN { namespace utils { class DetectionLayer; } }
namespace GOT { namespace segmentation { class ObjectProposal; } }

namespace SUN {
    namespace utils {
        namespace IO {

            /**
               * @brief Read content from file to eigen (double) matrix.
               * @param[in] filename
               * @param[in] rows
               * @param[in] cols
               * @param[in] mat_out
               * @return Sucess flag.
               * @author Aljosa (osep@vision.rwth-aachen.de)
               */
            bool ReadEigenMatrixFromTXT(const char *filename, Eigen::MatrixXd &mat_out);

            /**
               * @brief Creates directory.
               * @param[in] mat
               * @param[in] filename
               * @return Returns true upon successful creation of directory.
               * @author Aljosa (osep@vision.rwth-aachen.de)
               */
            bool MakeDir(const char *path);

            /**
           * @brief Read KITTI velodyne data.
           * @author Francis (engelmann@vision.rwth-aachen.de)
           */
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ReadLaserPointCloud(const std::string file_path);
            
        }
    }
}

#endif

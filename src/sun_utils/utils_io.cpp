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

#include "utils_io.h"

// Boost
#include <boost/filesystem.hpp>

// OpenCV
#include <opencv2/highgui/highgui.hpp>

namespace SUN {
    namespace utils {
        namespace IO {

            bool ReadEigenMatrixFromTXT(const char *filename, Eigen::MatrixXd &mat_out) {

                // General structure
                // 1. Read file contents into vector<double> and count number of lines
                // 2. Initialize matrix
                // 3. Put data in vector<double> into matrix

                std::ifstream input(filename);
                if (input.fail()) {
                    std::cerr << "ReadEigenMatrixFromTXT::Error: Can't Open file:'" << filename << "'." << std::endl;
                    mat_out = Eigen::MatrixXd(0,0);
                    return false;
                }
                std::string line;
                double d;

                std::vector<double> v;
                int n_rows = 0;
                while (getline(input, line)) {
                    ++n_rows;
                    std::stringstream input_line(line);
                    while (!input_line.eof()) {
                        input_line >> d;
                        v.push_back(d);
                    }
                }
                input.close();

                int n_cols = v.size()/n_rows;
                mat_out = Eigen::MatrixXd(n_rows,n_cols);

                for (int i=0; i<n_rows; i++)
                    for (int j=0; j<n_cols; j++)
                        mat_out(i,j) = v[i*n_cols + j];

                return true;
            }

            bool MakeDir(const char *path) {
                if (!path) {
                    return false;
                }

                boost::filesystem::path fpath(path);
                if (!boost::filesystem::exists(fpath)) {
                    boost::filesystem::path dir(fpath);
                    try {
                        boost::filesystem::create_directories(dir);
                    }
                    catch (boost::filesystem::filesystem_error e) {
                        std::cout << "Error: " << std::endl << e.what() << std::endl;
                        return false;
                    }
                }
                return true;
            }

            /*
             * Authors: Francis Engelmann (engelmann@vision.rwth-aachen.de), fixed by Aljosa Osep (osep@vision.rwth-aachen.de)
             */
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ReadLaserPointCloud(const std::string file_path) {

                // Allocate 4 MB buffer (only ~130*4*4 KB are needed)
                int32_t num = 1000000;

                FILE *stream;
                stream = fopen(file_path.c_str(),"rb");

                if (!stream) {
                    printf ("Could not Open velodyne scan: %s\r\n", file_path.c_str());
                    return nullptr; // Empty
                }

                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
                float *data = (float*)malloc(num*sizeof(float));

                // Pointers
                float *px = data+0;
                float *py = data+1;
                float *pz = data+2;
                float *pr = data+3;

                num = fread(data,sizeof(float),num,stream)/4;
                for (int32_t i=0; i<num; i++) {
                    pcl::PointXYZRGBA point;
                    point.x = *px;
                    point.y = *py;
                    point.z = *pz;

                    point.r = 255;
                    point.g = 0;
                    point.b = 0;

                    //double distance_to_velodyn = (point.x*point.x+point.y*point.y+point.z*point.z);
                    //double distance_threshold = 3; // distance to the car/velodyn in m. Within this radius we ignore points as they belong to the recording car
                    // Ignore points closer then threshold, they belong to car.
                    //if (distance_to_velodyn < distance_threshold*distance_threshold) continue;

                    point_cloud->points.push_back(point);
                    px+=4; py+=4; pz+=4; pr+=4;
                }

                free(data);

                fclose(stream);
                return point_cloud;
            }

        }
    }
}




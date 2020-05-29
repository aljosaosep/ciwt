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

#include "utils_kitti.h"

// Boost includes
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/trim.hpp>


namespace SUN {
    namespace utils {
        namespace KITTI {

            // -------------------------------------------------------------------------------
            // +++ Calibration Implementation +++
            // -------------------------------------------------------------------------------
            Calibration::Calibration() {
                //
            }

            Calibration::Calibration(const std::string &filename) {
                if (!this->Open(filename)) {
                    throw ("Could not Open calibration file: "+filename);
                }
            }

            Calibration::Calibration(const Calibration &calibration) {
                this->P_ = calibration.GetP();

                this->pos_cam_0_ = calibration.GetPosCam0();
                this->pos_cam_1_ = calibration.GetPosCam1();
                this->pos_cam_2_ = calibration.GetPosCam2();
                this->pos_cam_3_ = calibration.GetPosCam3();

                this->proj_cam_0_ = calibration.GetProjCam0();
                this->proj_cam_1_ = calibration.GetProjCam1();
                this->proj_cam_2_ = calibration.GetProjCam2();
                this->proj_cam_3_ = calibration.GetProjCam3();

                this->R_rect_ = calibration.getR_rect();

                this->Tr_imu_velo_ = calibration.getTr_imu_velo();
                this->Tr_velo_cam_ = calibration.getTr_velo_cam();

                this->baseline_ = std::sqrt( ((pos_cam_2_-pos_cam_3_).dot(pos_cam_2_-pos_cam_3_)) );
            }


            // The calib files in odometry and tracking differ, so two cases are needed...beeeerk
            bool Calibration::Open(const std::string &filename) {
                calib_file.open(filename.c_str());
                if (!calib_file.is_open()) {
                    std::cout << "ERROR: Could not Open calibration file: " << filename <<  std::endl;
                    return false;
                }
                // Read all lines of kitti calib file
                std::vector<std::string> calibLines;
                std::string line;
                while (getline(calib_file,line)) {
                    calibLines.push_back(line);
                }
                //calibrationFile_.close();
                // Split the line
                if (calibLines.size() == 2) { //
                    LabelsIO::ConvertLineToMatrix(calibLines.at(0), P_);
                    LabelsIO::ConvertLineToMatrix(calibLines.at(0), proj_cam_2_);
                    LabelsIO::ConvertLineToMatrix(calibLines.at(1), proj_cam_3_);
                } else { //
                    LabelsIO::ConvertLineToMatrix(calibLines.at(1), P_);
                    LabelsIO::ConvertLineToMatrix(calibLines.at(0), proj_cam_0_);
                    LabelsIO::ConvertLineToMatrix(calibLines.at(1), proj_cam_1_);
                    LabelsIO::ConvertLineToMatrix(calibLines.at(2), proj_cam_2_);
                    LabelsIO::ConvertLineToMatrix(calibLines.at(3), proj_cam_3_);
                    if (calibLines.size() == 7) {
                        LabelsIO::ConvertLineToMatrix(calibLines.at(4), R_rect_);
                        R_rect_(0,3) = 0.0f; R_rect_(1,3) = 0.0f; R_rect_(2,3) = 0.0f;
                        LabelsIO::ConvertLineToMatrix(calibLines.at(5), Tr_velo_cam_);
                        LabelsIO::ConvertLineToMatrix(calibLines.at(6), Tr_imu_velo_);
                    } else if (calibLines.size() == 5) { // This is the case for the odometry dataset.
                        LabelsIO::ConvertLineToMatrix(calibLines.at(4), Tr_velo_cam_);
                        R_rect_ = Eigen::Matrix<double,4,4>::Identity();
                    } else {
                        Tr_velo_cam_ = Eigen::Matrix<double,4,4>::Identity();
                        R_rect_ = Eigen::Matrix<double,4,4>::Identity();
                    }
                }

                this->computeCameraCenters();
                // Compute baselines as Euclidean-distance between corresponding camera-centers.
                this->baseline_ = sqrt( ((pos_cam_2_-pos_cam_3_).dot(pos_cam_2_-pos_cam_3_)) );
                return true;
            }

            double Calibration::b(void) const {
                // Compute baseline as Euclidean-distance between corresponding camera-centers.
                return baseline_;
            }

            void Calibration::computeCameraCenters() {
                // Physical order of the cameras: 2 0 3 1 (Camera 0 is reference)
                // Camera 0 and 1 are grayscale, 2 and 3 are color, left and right respectively.

                // Compute camera positions t using given projection P = K*[R|t]
                // Assuming that R = 0, correct?

                pos_cam_0_(2,0) = proj_cam_0_(2,3);
                pos_cam_0_(0,0) = (proj_cam_0_(0,3) - proj_cam_0_(0,2)*pos_cam_0_(2,0)) / proj_cam_0_(0,0);
                pos_cam_0_(1,0) = (proj_cam_0_(1,3) - proj_cam_0_(1,2)*pos_cam_0_(2,0)) / proj_cam_0_(1,1);

                pos_cam_1_(2,0) = proj_cam_1_(2,3);
                pos_cam_1_(0,0) = (proj_cam_1_(0,3) - proj_cam_1_(0,2)*pos_cam_1_(2,0)) / proj_cam_1_(0,0);
                pos_cam_1_(1,0) = (proj_cam_1_(1,3) - proj_cam_1_(1,2)*pos_cam_1_(2,0)) / proj_cam_1_(1,1);

                pos_cam_2_(2,0) = proj_cam_2_(2,3);
                pos_cam_2_(0,0) = (proj_cam_2_(0,3) - proj_cam_2_(0,2)*pos_cam_2_(2,0)) / proj_cam_2_(0,0);
                pos_cam_2_(1,0) = (proj_cam_2_(1,3) - proj_cam_2_(1,2)*pos_cam_2_(2,0)) / proj_cam_2_(1,1);

                pos_cam_3_(2,0) = proj_cam_3_(2,3);
                pos_cam_3_(0,0) = (proj_cam_3_(0,3) - proj_cam_3_(0,2)*pos_cam_3_(2,0)) / proj_cam_3_(0,0);
                pos_cam_3_(1,0) = (proj_cam_3_(1,3) - proj_cam_3_(1,2)*pos_cam_3_(2,0)) / proj_cam_3_(1,1);
            }

            Eigen::Matrix<double, 4, 4> Calibration::GetTr_cam0_cam2() const {
                Eigen::Matrix<double,4,4> m = Eigen::Matrix<double,4,4>::Identity();;
                m.block<3,1>(0,3) = pos_cam_2_.block<3,1>(0,0);
                return m;
            }

            // -------------------------------------------------------------------------------
            // +++ LabelsIO Implementation +++
            // -------------------------------------------------------------------------------
            LabelsIO::LabelsIO()
            {

            }

//            int LabelsIO::ReadLabels(const std::string &filename, std::vector<TrackingLabel> &labels) {
//                std::ifstream file(filename.c_str(), std::ios::in);
//                if (file.fail()) {
//                    std::cout << "Could not read labels file " << filename << std::endl;
//                    return -1;
//                }
//
//                unsigned int numObjects = 0;
//
//                //read line by line
//                std::string line;
//                while (getline (file,line)) {
//                    std::vector<std::string> strings(17); // Each line has 30 space-seperated entries
//                    boost::split(strings,line,boost::is_any_of(" "));
//                    TrackingLabel label;
//                    label.frame = atoi( (strings.at(0)).c_str() );
//                    label.trackId = atoi( (strings.at(1)).c_str() );
//                    if (label.trackId+1 > numObjects) {
//                        numObjects=label.trackId+1;
//                    }
//                    std::string type = strings.at(2);
//                    if (type.compare("Car")==0) {
//                        label.type = SUN::shared_types::CAR;
//                    } else if (type.compare("Van")==0) {
//                        label.type = SUN::shared_types::VAN;
//                    } else if (type.compare("Truck")==0) {
//                        label.type = SUN::shared_types::TRUCK;
//                    } else if (type.compare("Pedestrian")==0) {
//                        label.type = SUN::shared_types::PEDESTRIAN;
//                    } else if (type.compare("Person_sitting")==0) {
//                        label.type = SUN::shared_types::PERSON_SITTING;
//                    } else if (type.compare("Cyclist")==0) {
//                        label.type = SUN::shared_types::CYCLIST;
//                    } else if (type.compare("Tram")==0) {
//                        label.type = SUN::shared_types::TRAM;
//                    } else if (type.compare("Misc")==0) {
//                        label.type = SUN::shared_types::MISC;
//                    } else if (type.compare("DontCare")==0) {
//                        label.type = SUN::shared_types::DONT_CARE;
//                    } else if (type.compare("UNKNOWN")==0) {
//                        label.type = SUN::shared_types::UNKNOWN_TYPE;
//                    } else {
//                        label.type = SUN::shared_types::MISC;
//                    }
//                    label.truncated = atof( (strings.at(3)).c_str() );
//                    label.occluded = static_cast<LabelOcclusion>(atoi( (strings.at(4)).c_str() ));
//                    label.alpha = atof( (strings.at(5)).c_str() );
//                    label.boundingBox2D[0] = atof( (strings.at(6)).c_str() );
//                    label.boundingBox2D[1] = atof( (strings.at(7)).c_str() );
//                    label.boundingBox2D[2] = atof( (strings.at(8)).c_str() );
//                    label.boundingBox2D[3] = atof( (strings.at(9)).c_str() );
//                    label.dimensions[0] = atof( (strings.at(10)).c_str() );
//                    label.dimensions[1] = atof( (strings.at(11)).c_str() );
//                    label.dimensions[2] = atof( (strings.at(12)).c_str() );
//                    label.location[0] = atof( (strings.at(13)).c_str() );
//                    label.location[1] = atof( (strings.at(14)).c_str() );
//                    label.location[2] = atof( (strings.at(15)).c_str() );
//                    label.rotationY = atof( (strings.at(16)).c_str() );
//
//                    if (strings.size() >= 18) // This is a bit hacky solution ...
//                        label.score = atof( (strings.at(17)).c_str() );
//                    else
//                        label.score = 0.0;
//
//                    labels.push_back(label);
//                }
//                return numObjects;
//            }


            int LabelsIO::ReadLabels(const std::string &filename, std::vector<TrackingLabel> &labels) {

                labels.clear();

                std::ifstream file(filename.c_str(), std::ios::in);
                if (file.fail()) {
                    std::cout << "Could not read labels file " << filename << std::endl;
                    return -1;
                }

                unsigned int numObjects = 0;

                //read line by line
                std::string line;
                while (getline (file,line)) {
                    std::vector<std::string> strings(17); // Each line has 30 space-seperated entries
                    boost::split(strings,line,boost::is_any_of(" "));
                    TrackingLabel label;

                    // Get frame from filename!
                    auto frame_str = filename.substr(7, 2);
                    label.frame = atoi(frame_str.c_str());

                    label.trackId = -1; //atoi( (strings.at(1)).c_str() );
                    if (label.trackId+1 > numObjects) {
                        numObjects=label.trackId+1;
                    }
                    std::string type = strings.at(0);
                    if (type.compare("Car")==0) {
                        label.type = SUN::shared_types::CAR;
                    } else if (type.compare("Van")==0) {
                        label.type = SUN::shared_types::VAN;
                    } else if (type.compare("Truck")==0) {
                        label.type = SUN::shared_types::TRUCK;
                    } else if (type.compare("Pedestrian")==0) {
                        label.type = SUN::shared_types::PEDESTRIAN;
                    } else if (type.compare("Person_sitting")==0) {
                        label.type = SUN::shared_types::PERSON_SITTING;
                    } else if (type.compare("Cyclist")==0) {
                        label.type = SUN::shared_types::CYCLIST;
                    } else if (type.compare("Tram")==0) {
                        label.type = SUN::shared_types::TRAM;
                    } else if (type.compare("Misc")==0) {
                        label.type = SUN::shared_types::MISC;
                    } else if (type.compare("DontCare")==0) {
                        label.type = SUN::shared_types::DONT_CARE;
                    } else if (type.compare("UNKNOWN")==0) {
                        label.type = SUN::shared_types::UNKNOWN_TYPE;
                    } else {
                        label.type = SUN::shared_types::MISC;
                    }
                    label.truncated = atof( (strings.at(1)).c_str() );
                    label.occluded = static_cast<LabelOcclusion>(atoi( (strings.at(2)).c_str() ));
                    label.alpha = atof( (strings.at(3)).c_str() );
                    label.boundingBox2D[0] = atof( (strings.at(4)).c_str() );
                    label.boundingBox2D[1] = atof( (strings.at(5)).c_str() );
                    label.boundingBox2D[2] = atof( (strings.at(6)).c_str() );
                    label.boundingBox2D[3] = atof( (strings.at(7)).c_str() );
                    label.dimensions[0] = atof( (strings.at(8)).c_str() );
                    label.dimensions[1] = atof( (strings.at(9)).c_str() );
                    label.dimensions[2] = atof( (strings.at(10)).c_str() );
                    label.location[0] = atof( (strings.at(11)).c_str() );
                    label.location[1] = atof( (strings.at(12)).c_str() );
                    label.location[2] = atof( (strings.at(13)).c_str() );
                    label.rotationY = atof( (strings.at(14)).c_str() );
                    label.score = atof( (strings.at(15)).c_str() );


                    labels.push_back(label);
                }
                return numObjects;
            }

            int LabelsIO::WriteLabels(std::vector<TrackingLabel> &labels, const std::string &filename) {

                std::ofstream file(filename.c_str(), std::ofstream::app);
                if (!file.is_open()) {
                    std::cout << "Could not write labels file: " << filename << std::endl;
                    return -1;
                }

                LabelsIO sun_io;
                unsigned int numObjects = 0;

                for (const auto &label:labels) {
                    // Generate label string based on the value of LabelType type;
                    std::string category_string = "UNKNOWN";
                    if (label.type==SUN::shared_types::CategoryTypeKITTI::CAR) {
                        category_string = "Car";
                    } else if (label.type==SUN::shared_types::CategoryTypeKITTI::VAN) {
                        category_string = "Van";
                    } else if (label.type==SUN::shared_types::CategoryTypeKITTI::TRUCK) {
                        category_string = "Truck";
                    } else if (label.type==SUN::shared_types::CategoryTypeKITTI::PEDESTRIAN) {
                        category_string = "Pedestrian";
                    } else if (label.type==SUN::shared_types::CategoryTypeKITTI::PERSON_SITTING) {
                        category_string = "Person_Sitting";
                    } else if (label.type==SUN::shared_types::CategoryTypeKITTI::CYCLIST) {
                        category_string = "Cyclist";
                    } else if (label.type==SUN::shared_types::CategoryTypeKITTI::TRAM) {
                        category_string = "Tram";
                    } else if (label.type==SUN::shared_types::CategoryTypeKITTI::MISC) {
                        category_string = "Misc";
                    } else if (label.type==SUN::shared_types::CategoryTypeKITTI::DONT_CARE) {
                        category_string = "DontCare";
                    }

                    sun_io.WriteStringToFile(std::to_string(label.frame), file);
                    sun_io.WriteStringToFile(std::to_string(label.trackId), file);
                    sun_io.WriteStringToFile(category_string, file);
                    sun_io.WriteStringToFile(std::to_string(label.truncated), file);
                    sun_io.WriteStringToFile(std::to_string(label.occluded), file);
                    sun_io.WriteStringToFile(std::to_string(label.alpha), file);
                    sun_io.WriteStringToFile(std::to_string(label.boundingBox2D[0]), file);
                    sun_io.WriteStringToFile(std::to_string(label.boundingBox2D[1]), file);
                    sun_io.WriteStringToFile(std::to_string(label.boundingBox2D[2]), file);
                    sun_io.WriteStringToFile(std::to_string(label.boundingBox2D[3]), file);
                    sun_io.WriteStringToFile(std::to_string(label.dimensions[0]), file);
                    sun_io.WriteStringToFile(std::to_string(label.dimensions[1]), file);
                    sun_io.WriteStringToFile(std::to_string(label.dimensions[2]), file);
                    sun_io.WriteStringToFile(std::to_string(label.location[0]), file);
                    sun_io.WriteStringToFile(std::to_string(label.location[1]), file);
                    sun_io.WriteStringToFile(std::to_string(label.location[2]), file);
                    sun_io.WriteStringToFile(std::to_string(label.rotationY), file);
                    sun_io.WriteStringToFile(std::to_string(label.score), file);

                    file << "\n";

                    if (label.trackId+1 > numObjects) {
                        numObjects=label.trackId+1;
                    }
                }
                return numObjects;
            }

            void LabelsIO::ConvertLineToMatrix(std::string &line, Eigen::Matrix<double, 3,4> &matrix) {
                std::vector<std::string> strings;//(3*4+1);
                boost::trim(line);
                boost::split(strings,line,boost::is_any_of(" "));
                size_t cols,rows;
                unsigned char offset = 1;
                if (strings.size() == 13) {
                    cols = 4; rows = 3;
                } else if (strings.size() == 12) {
                    cols = 4; rows = 3;
                    offset = 0;
                } else if (strings.size() == 10) {
                    cols = 3; rows = 3;
                } else {
                    std::cout << "Error while converting line to matrix." << std::endl;
                    std::cout << "Using size " << strings.size() << std::endl;
                }
                for (size_t i=0; i<rows; i++) {
                    for (size_t j=0; j<cols; j++) {
                        size_t index = (cols*i)+j + offset; // + 1 is to ignore the first column which indicates the name
                        matrix(i,j) = atof( strings.at(index).c_str() );
                    }
                }
            }

            void LabelsIO::ConvertLineToMatrix(std::string &line, Eigen::Matrix<double, 4, 4> &matrix) {
                matrix = Eigen::MatrixXd::Identity(4,4);
                Eigen::Matrix<double,3,4> tmp;
                LabelsIO::ConvertLineToMatrix(line, tmp);
                matrix.block<3,4>(0,0) = tmp;
            }

            void LabelsIO::WriteStringToFile(std::string text, std::ofstream &file) {
                file << text;
                file << " ";
            }
        }
    }
}
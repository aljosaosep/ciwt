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

#include "datasets_dirty_utils.h"

// boost
#include <boost/filesystem.hpp>

// utils
#include "sun_utils/utils_io.h"
#include "sun_utils/utils_kitti.h"
#include "sun_utils/utils_pointcloud.h"
#include "sun_utils/ground_model.h"
#include "sun_utils/utils_observations.h"
#include "sun_utils/shared_types.h"


// Elas
#include <libelas/elas.h>

#define MAX_PATH_LEN 500

namespace SUN {
    namespace utils {
        namespace dirty {


            // -------------------------------------------------------------------------------
            // +++ UTILS  +++
            // -------------------------------------------------------------------------------
            void ComputeDisparityElas(const cv::Mat &image_left, const cv::Mat &image_right,
                                        SUN::DisparityMap &disparity_left, SUN::DisparityMap &disparity_right) {

                cv::Mat grayscale_left, grayscale_right;
                cv::cvtColor(image_left, grayscale_left, cv::COLOR_BGR2GRAY);
                cv::cvtColor(image_right, grayscale_right, cv::COLOR_BGR2GRAY);

                // get image width and height
                int32_t width  = grayscale_left.cols;
                int32_t height = grayscale_right.rows;

                // allocate memory for disparity images
                const int32_t dims[3] = {width,height,width}; // bytes per line = width
                float* D1_data = (float*)malloc(width*height*sizeof(float));
                float* D2_data = (float*)malloc(width*height*sizeof(float));

                // process
                libelas::Elas::parameters param;

                // Enable these params if you want output disp. maps to be the same as ones used for KITTI eval.
                param.postprocess_only_left = false;
                //param.add_corners = 1;
                //param.match_texture = 0;

                std::vector<libelas::Elas::support_pt> support_points;
                libelas::Elas elas(param);
                elas.process(grayscale_left.data, grayscale_right.data, D1_data, D2_data, dims, support_points);

                disparity_left = SUN::DisparityMap(D1_data, width, height);
                disparity_right = SUN::DisparityMap(D2_data, width, height);

                free(D1_data);
                free(D2_data);
            }


            // -------------------------------------------------------------------------------
            // +++ DATASET ASSISTANT IMPLEMENTATION +++
            // -------------------------------------------------------------------------------

            DatasetAssitantDirty::DatasetAssitantDirty(const po::variables_map &config_variables_map) {
                this->variables_map_ = config_variables_map;
                stereo_baseline_ = -1;
            }

            bool DatasetAssitantDirty::RequestDisparity(int frame, bool save_if_not_avalible) {

                if (!variables_map_.count("left_disparity_path")) {
                    printf("Error, disparity map path option not specified: left_disparity_path.\r\n");
                    return false;
                }

                char left_disparity_map_buff[MAX_PATH_LEN];
                snprintf(left_disparity_map_buff, MAX_PATH_LEN, this->variables_map_["left_disparity_path"].as<std::string>().c_str(), frame);

                // Try to read-in disparity
                disparity_map_.Read(left_disparity_map_buff);

                // No precomputed disparity, see if we have left and right images. If yes, run matching.
                if (disparity_map_.mat().data == nullptr) {
                    printf("Could not load disparity map: %s, running ELAS ...\r\n", left_disparity_map_buff);

                    if (this->left_image_.data==nullptr || this->right_image_.data==nullptr) {
                        printf("Left and right images not available, aborting stereo.\r\n");
                        return false;
                    }

                    SUN::DisparityMap disparity_left, disparity_right;
                    ComputeDisparityElas(this->left_image_, this->right_image_, disparity_left, disparity_right);
                    disparity_map_ = disparity_left;

                    if (save_if_not_avalible) {
                        printf("Saving disparity map to: %s.\r\n", left_disparity_map_buff);
                        boost::filesystem::path prop_path(left_disparity_map_buff);
                        boost::filesystem::path prop_dir = prop_path.parent_path();
                        SUN::utils::IO::MakeDir(prop_dir.c_str());
                        disparity_left.WriteDisparityMap(std::string(left_disparity_map_buff));
                    }
                }

                return true;
            }

            std::vector<SUN::utils::KITTI::TrackingLabel> ReadDetections(int current_frame,
                    const po::variables_map &variables_map) {

                SUN::utils::KITTI::LabelsIO sun_io;
                std::string detections_path = variables_map["detections_path"].as<std::string>();

                char det_buff[500];
                snprintf(det_buff, 500, detections_path.c_str(), current_frame);

                std::vector<SUN::utils::KITTI::TrackingLabel> detlabels;
                sun_io.ReadLabels(std::string(det_buff), detlabels);
                return detlabels;
            }

            bool DatasetAssitantDirty::LoadData(int current_frame, const std::string dataset_string) {
                std::string dataset_str_lower = dataset_string;
                std::transform(dataset_str_lower.begin(), dataset_str_lower.end(), dataset_str_lower.begin(), ::tolower);

                bool status = false;
                if (dataset_string=="kitti")
                    status = this->LoadData__KITTI(current_frame);
//                else if (dataset_string=="kitti_disparity")
//                    status = this->LoadData__KITTI__DISPARITY(current_frame);
//                else if (dataset_string=="schipool")
//                    status = this->LoadData__SCHIPOOL(current_frame);
//                else if (dataset_string=="rwth")
//                    status = this->LoadData__RWTH(current_frame);
                else
                    std::cout << "DatasetAssitantDirty error: no such dataset: " << dataset_string << std::endl;

                return status;
            }

            bool DatasetAssitantDirty::LoadData__KITTI(int current_frame) {

                // -------------------------------------------------------------------------------
                // +++ ABSOLUTELY REQUIRED DATA +++
                // -------------------------------------------------------------------------------
                char left_image_path_buff[MAX_PATH_LEN];
                snprintf(left_image_path_buff, MAX_PATH_LEN, this->variables_map_["left_image_path"].as<std::string>().c_str(), current_frame);

                /// KITTI camera calibration
                SUN::utils::KITTI::Calibration calibration;
                const std::string calib_path = this->variables_map_["calib_path"].as<std::string>();
                if (!calibration.Open(calib_path)) {
                    printf("DatasetAssitantDirty error: Can't Open calibration file: %s\r\n", calib_path.c_str());
                    return false;
                }

                /// Image data
                left_image_ = cv::imread(left_image_path_buff, cv::IMREAD_COLOR);
                if (left_image_.data == nullptr) {
                    printf("DatasetAssitantDirty error: could not load image: %s\r\n", left_image_path_buff);
                    return false;
                }

                /// Init camera and ground-model
                left_camera_.init(calibration.GetProjCam2(), Eigen::Matrix4d::Identity(), left_image_.cols, left_image_.rows);
                right_camera_.init(calibration.GetProjCam3(), Eigen::Matrix4d::Identity(), left_image_.cols, left_image_.rows);
                stereo_baseline_ = calibration.b();

                // -------------------------------------------------------------------------------
                // +++ OPTIONAL STUFF +++
                // -------------------------------------------------------------------------------

                /// Right image
                if (this->variables_map_.count("right_image_path")) {
                    char right_image_path_buff[MAX_PATH_LEN];
                    snprintf(right_image_path_buff, MAX_PATH_LEN, this->variables_map_["right_image_path"].as<std::string>().c_str(), current_frame);
                    right_image_ = cv::imread(right_image_path_buff, cv::IMREAD_COLOR);

                    if (right_image_.data == nullptr) {
                        printf("DatasetAssitantDirty error: could not load image: %s\r\n", left_image_path_buff);
                        return false;
                    }
                }

                /// Disparity map
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud_ptr = nullptr;
                if (this->variables_map_.count("left_disparity_path")) {

                    if (!(this->RequestDisparity(current_frame, true))) {
                        printf("DatasetAssitantDirty error: RequestDisparity failed!\r\n");
                        return false;
                    }

                    char left_disparity_map_buff[MAX_PATH_LEN];
                    snprintf(left_disparity_map_buff, MAX_PATH_LEN, this->variables_map_["left_disparity_path"].as<std::string>().c_str(), current_frame);

                    disparity_map_.Read(left_disparity_map_buff);

                    if (disparity_map_.mat().data == nullptr) {
                        printf("DatasetAssitantDirty error: could not load disparity map: %s\r\n", left_disparity_map_buff);
                        return false;
                    }

                    /// Compute point cloud. Note, that this point cloud is in camera space (current frame).
                    left_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
                    Eigen::Matrix<double, 4, 4> identity_matrix_4 = Eigen::MatrixXd::Identity(4, 4);
                    SUN::utils::pointcloud::ConvertDisparityMapToPointCloud(disparity_map_.mat(), left_image_,
                                                                            calibration.c_u(), calibration.c_v(), calibration.f(), calibration.b(),
                                                                            Eigen::Matrix4d::Identity(), true, left_point_cloud_);
                    point_cloud_ptr = left_point_cloud_;
                }



                /// Load whole-sequence detections, but only once!
//                if (this->variables_map_.count("detections_path")) {
//                    if (this->kitti_detections_full_sequence_.size() <= 0) {
//                        std::cout << "Loading KITTI detections for the whole sequence ..." << std::endl;
//                        SUN::utils::KITTI::LabelsIO sun_io;
//                        sun_io.ReadLabels(this->variables_map_["detections_path"].as<std::string>(), kitti_detections_full_sequence_);
//                    }
//                }

                parsed_det_ = ReadDetections(current_frame, variables_map_);

                /// LiDAR
                // Note: If velodyne scan is missing, don't return false. There are some velodyne scans actually missing in the dataset.
                if (this->variables_map_.count("velodyne_path")) {
                    char velodyne_buff[500];
                    snprintf(velodyne_buff, MAX_PATH_LEN, this->variables_map_["velodyne_path"].as<std::string>().c_str(), current_frame);
                    auto velodyne_cloud = SUN::utils::IO::ReadLaserPointCloud(std::string(velodyne_buff));
                    if (velodyne_cloud!=nullptr) {
                        // There are actually some velodyne scans missing ... ignore those.
                        Eigen::Matrix4d Tr_cam0_cam2 = calibration.GetTr_cam0_cam2(); // Translation cam0 -> cam2
                        Eigen::Matrix<double, 4, 4> Tr_laser_cam2 = Tr_cam0_cam2 * calibration.getR_rect() * calibration.getTr_velo_cam();
                        left_point_cloud_velodyne_ = SUN::utils::pointcloud::RawLiDARCloudToImageAlignedAndOrganized(velodyne_cloud, Tr_laser_cam2, left_image_, left_camera_);
                    }

                    point_cloud_ptr = left_point_cloud_velodyne_;
                }


                /// Ground-plane
                bool got_gp = false;
                std::shared_ptr<SUN::utils::PlanarGroundModel> planar_ground_model(new SUN::utils::PlanarGroundModel);

                if (this->variables_map_.count("ground_plane_path")) {
                    char ground_plane_buff[MAX_PATH_LEN];
                    snprintf(ground_plane_buff, MAX_PATH_LEN, this->variables_map_["ground_plane_path"].as<std::string>().c_str(), current_frame);
                    Eigen::MatrixXd gp_tmp;
                    if (SUN::utils::IO::ReadEigenMatrixFromTXT(ground_plane_buff, gp_tmp)) {
                        ground_plane_ = gp_tmp.row(0).head<4>();
                        planar_ground_model->set_plane_params(ground_plane_);
                        got_gp = true;
                    }
                }

                if (!got_gp) {
                    // Fit plane
                    printf ("Could not load ground-plane parameters, fitting plane of-the-fly ...\r\n");
                    planar_ground_model->FitModel(point_cloud_ptr, 1.4);
                    // TODO: check if fitting was successful!
                }

                /// Link it to the camera obj's
                left_camera_.set_ground_model(planar_ground_model);
                right_camera_.set_ground_model(planar_ground_model);

//                /// Detections
//                if (this->variables_map_.count("detections_path")) {
//                    // object_detections_ = GetDetectionsKITTI(current_frame, left_camera_, right_camera_, variables_map_,
//                    //        this->kitti_detections_full_sequence_);
//                    object_detections_ = GetDetections3DOP(current_frame, left_camera_, calibration, variables_map_);
//                }

                /// Velocity estimates (from scene-flow)
                if (this->variables_map_.count("flow_map_path")) {
                    char buff[500];
                    snprintf(buff, 500, this->variables_map_["flow_map_path"].as<std::string>().c_str(), current_frame);
                    this->velocity_map_ = cv::imread(buff, cv::IMREAD_UNCHANGED);
                    if (this->velocity_map_.data == nullptr) {
                        printf("Error, could not load flow-file: %s\r\n", buff);
                        return false;
                    }
                }

                return true;
            }

        }
    }
}
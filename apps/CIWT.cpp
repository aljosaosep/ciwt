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

// C
#include <ctime>

// std
#include <iostream>
#include <memory>
#include <algorithm>
#include <list>
#include <chrono>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

// boost
#include <boost/archive/binary_iarchive.hpp>

// scene segmentation
#include <scene_segmentation/scene_segmentation.h>
#include <scene_segmentation/utils_segmentation.h>
#include <scene_segmentation/multi_scale_quickshift.h>
#include <scene_segmentation/parameters_gop3D.h>

// tracking
#include <tracking/visualization.h>
#include <tracking/utils_tracking.h>
#include <src/sun_utils/detection.h>
#include <tracking/category_filter.h>

// utils
#include "utils_io.h"
#include "utils_visualization.h"
#include "utils_pointcloud.h"
#include "ground_model.h"
#include "utils_observations.h"
#include "datasets_dirty_utils.h"
#include "utils_bounding_box.h"
#include "utils_common.h"
#include "utils_filtering.h"

// CIWT
#include "CIWT/parameters_CIWT.h"
#include "CIWT/CIWT_tracker.h"
#include "CIWT/observation_fusion.h"
#include "CIWT/potential_functions.h"

#define MAX_PATH_LEN 500

// For convenience.
namespace po = boost::program_options;
typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;

namespace CIWTApp {
    // We need those for the 3d visualizer thread.
    bool visualization_3d_update_flag;
    boost::mutex visualization_3d_update_mutex;
    PointCloudRGBA::Ptr visualization_3d_point_cloud;
    GOT::tracking::HypothesesVector visualization_3d_tracking_hypotheses;
    GOT::tracking::HypothesesVector visualization_3d_tracking_terminated_hypotheses;
    std::vector<GOT::tracking::Observation> visualization_observations;
    std::vector<GOT::segmentation::ObjectProposal> visualization_3d_proposals;
    SUN::utils::Camera visualization_3d_left_camera;
    GOT::tracking::Visualizer tracking_visualizer;

    // Paths
    std::string output_dir;
    std::string proposals_path;
    std::string tracking_mode_str;
    std::string viewer_3D_output_path;
    std::string config_parameters_file;
    std::string sequence_name;
    std::string dataset_name;

    // Application data.
    bool show_visualization_2d;
    bool show_visualization_3d;

    int start_frame;
    int end_frame;
    int debug_level;
    bool run_tracker;

    Eigen::Matrix4d egomotion = Eigen::Matrix4d::Identity();

    void VisualizeScene3D() {
        // Set up the visualizer.
        pcl::visualization::PCLVisualizer viewer("3D Scene Viewer");
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgbHandler(visualization_3d_point_cloud);

        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
        viewer.setBackgroundColor (0.0, 0.0, 0.0);
        viewer.setCameraPosition(-0.643694,-1.45647,-4.11917, //camera optical center position
                                 -0.0132233,-0.999894,0.0060805,  //Look-at-position
                                 0, -1, 0);   //camera up vector

        viewer.setCameraFieldOfView(0.8575);
        viewer.setCameraClipDistances(0.0001, 900);
        viewer.setSize(2000, 800);

        // Visualizer main loop.
        while (!viewer.wasStopped ()) {
            viewer.spinOnce(100);

            // Get lock on the boolean update and check if cloud was updated.
            boost::mutex::scoped_lock updateLock(visualization_3d_update_mutex);
            if (visualization_3d_update_flag) {
                viewer.removeAllShapes();
                viewer.removeAllPointClouds();

                // Show point-cloud.
                if(!viewer.updatePointCloud(visualization_3d_point_cloud, rgbHandler, "sample cloud")) {
                    viewer.addPointCloud(visualization_3d_point_cloud, rgbHandler, "sample cloud");
                    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "sample cloud");
                    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.7, "sample cloud");
                }

                // Settings
                bool draw_observations = false;
                bool draw_tracked_objects = true;
                bool draw_terminated_hypos = false;
                bool mark_only_proposal_region = false;
                bool draw_proposals = false;

                if (draw_proposals) {
                    // Render 3D bounding boxes.
                    for (int i=0; i<visualization_3d_proposals.size(); i++) {
                        // 3D bounding box: [centroid_x centroid_y centroid_z width height length q.w q.x q.y q.z]
                        auto &proposal = visualization_3d_proposals.at(i);
                        auto bbox_3d = proposal.bounding_box_3d();
                        std::string id = "bb_3d_" + std::to_string(i);
                        SUN::utils::visualization::RenderBoundingBox3D(viewer, bbox_3d, 0.0, 0.0, 1.0, id);
                    }
                }

                if (draw_observations) {
                    // Draw observations
                    for (int i=0; i<visualization_observations.size(); i++) {
                        const auto &obs = visualization_observations.at(i);

                        uint8_t r, g, b;
                        SUN::utils::visualization::GenerateColor(i,r,g,b);
                        double r_ = static_cast<double> (r) / 255.0;
                        double g_ = static_cast<double> (g) / 255.0;
                        double b_ = static_cast<double> (b) / 255.0;
                        PointCloudRGBA::Ptr seg_cloud(new PointCloudRGBA);

                        if (mark_only_proposal_region) {
                            // Draw only 'associated' detection part.
                            pcl::PointIndices inds;
                            inds.indices = obs.pointcloud_indices();
                            pcl::copyPointCloud(*visualization_3d_point_cloud, inds, *seg_cloud);
                        }
                        else {
                            // For ICRA'17 paper visualization: full-projection of bounding-box to point-cloud.
                            Eigen::Vector4d det_bbox_2d = obs.bounding_box_2d();
                            for (int i=det_bbox_2d[0]; i<det_bbox_2d[2]+det_bbox_2d[0]; i++) {
                                for (int j=det_bbox_2d[1]; j<det_bbox_2d[3]+det_bbox_2d[1]; j++) {
                                    pcl::PointXYZRGBA p = visualization_3d_point_cloud->at(i,j);
                                    if (!std::isnan(p.x)) {
                                        p.r = r; p.b = b; p.g = g;
                                        seg_cloud->push_back(p);
                                    }
                                }
                            }

                            seg_cloud->height = 1;
                            seg_cloud->width = seg_cloud->size();
                        }
                        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> seg_handler(seg_cloud);

                        auto seg_cloud_id = "seg_"+std::to_string(i);
                        if(!viewer.updatePointCloud(seg_cloud, seg_handler, seg_cloud_id)) {
                            viewer.addPointCloud(seg_cloud, seg_handler, seg_cloud_id);
                            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, seg_cloud_id);
                            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, seg_cloud_id);
                            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, r_, g_, b_, seg_cloud_id);
                        }

                    }
                }

                if (draw_tracked_objects) {
                    // Draw active hypotheses.
                    for (int i=0; i<visualization_3d_tracking_hypotheses.size(); i++) {
                        const auto &hypo = visualization_3d_tracking_hypotheses.at(i);
                        double r,g,b;
                        tracking_visualizer.GetColor(hypo.id(), b,g,r);

                        if (hypo.bounding_boxes_3d().size()>0) {
                            std::string id = "hypo_bbox_3d_" + std::to_string(i);
                            tracking_visualizer.RenderHypo3D(viewer, hypo, CIWTApp::visualization_3d_left_camera, 0);
                        }

                        tracking_visualizer.RenderTrajectory(hypo, CIWTApp::visualization_3d_left_camera, ("hypo_traj_3d_" + std::to_string(i)), r,g,b, viewer);
                    }

                    // Draw terminated hypotheses.
                    if (draw_terminated_hypos) {
                        for (int i=0; i<visualization_3d_tracking_terminated_hypotheses.size(); i++) {
                            const auto &hypo = visualization_3d_tracking_terminated_hypotheses.at(i);
                            if (hypo.bounding_boxes_3d().size() > 0) {
                                std::string id = "hypo_term_bbox_3d_" + std::to_string(i);
                                SUN::utils::visualization::RenderBoundingBox3D(viewer, hypo.bounding_boxes_3d().back(), 0.0, 0.0, 1.0, id, 0);
                            }
                            tracking_visualizer.RenderTrajectory(hypo, CIWTApp::visualization_3d_left_camera, ("hypo_term_traj_3d_" + std::to_string(i)), 0.0, 0.0, 1.0, viewer);
                        }
                    }
                }

                if (debug_level>=3)
                    viewer.saveScreenshot(viewer_3D_output_path);

                visualization_3d_update_flag = false;
                viewer.setCameraClipDistances(0.0001, 900);
            }
            updateLock.unlock();
        }
    }

    bool ParseCommandArguments(const int argc, const char **argv, po::variables_map &config_variables_map) {
        po::options_description cmdline_options;
        try {
            std::string config_file;

            // The very basic options, can be only specified via cmd args
            po::options_description generic_options("Command line options:");
            generic_options.add_options()
                    ("help", "Produce help message")
                    ("config", po::value<std::string>(&config_file), "Config file path.")
                    ("config_parameters", po::value<std::string>(&config_parameters_file), "Config file path (parameters only!).")
                    ("show_visualization_3d", po::bool_switch(&show_visualization_3d)->default_value(false), "Show 3D visualization.")
                    ("show_visualization_2d", po::bool_switch(&show_visualization_2d)->default_value(false), "Show 2D visualization.")
                    ("run_tracker", po::value<bool>(&run_tracker)->default_value(true), "Should run tracker, or should not?")
                    ("debug_level", po::value<int>(&debug_level)->default_value(0), "Debug level")
                    ;

            // General app options (can be spec. in config or via cmd args)
            po::options_description config_options("Config options");
            config_options.add_options()
                    // Paths
                    ("left_image_path", po::value<std::string>(), "Image (left) path")
                    ("right_image_path", po::value<std::string>(), "Image (right) path")
                    ("left_disparity_path", po::value<std::string>(), "Disparity (left) path")
                    ("calib_path", po::value<std::string>(), "Camera calibration path (currently supported: kitti)")
                    ("object_proposal_path", po::value<std::string>(&proposals_path)->default_value(""), "Proposals path")
                    ("ground_plane_path", po::value<std::string>(), "Ground-plane params.")
                    ("detections_path", po::value<std::string>()->default_value(""), "Resources path.")
                    ("flow_map_path", po::value<std::string>(), "Path to flow dir.")
                    ("output_dir", po::value<std::string>(&output_dir), "Output path")
                    ("start_frame", po::value<int>(&start_frame)->default_value(0), "Starting frame")
                    ("end_frame", po::value<int>(&end_frame)->default_value(10000), "Last frame")
                    ("tracking_mode", po::value<std::string>(&tracking_mode_str)->default_value("detection"), "Tracking mode: detection, detection_shape, detection_3DOP.")
                    ("sequence_name", po::value<std::string>(&sequence_name)->default_value("default_sequence_name"), "Name of the sequence")
                    ("dataset_name", po::value<std::string>(&dataset_name)->default_value("kitti"), "Name of the sequence")
                    ;

            // Add parameter options (def. in parameters_CIWT.h)
            po::options_description parameter_options("Parameters:");
            InitParameters(parameter_options);

            // Parameters for the 3D proposals
            po::options_description parameters_proposals("Parameters-proposals:");
            GOP3D::InitParameters(parameters_proposals);

            cmdline_options.add(generic_options);
            cmdline_options.add(config_options);
            cmdline_options.add(parameter_options);
            cmdline_options.add(parameters_proposals);

            store(po::command_line_parser(argc, argv).options(cmdline_options).run(), config_variables_map);

            if (config_variables_map.count("help")) {
                std::cout << cmdline_options << endl;
                return false;
            }

            notify(config_variables_map);

            // "generic" config
            if (config_variables_map.count("config")) {
                std::ifstream ifs(config_file.c_str());
                if (!ifs.is_open()) {
                    std::cout << "Can not Open config file: " << config_file << "\n";
                    return false;
                } else {
                    store(parse_config_file(ifs, cmdline_options), config_variables_map);
                    notify(config_variables_map);
                }
            }

            if (config_variables_map.count("config_parameters")) {
                // "parameter" config
                std::ifstream ifs_param(config_parameters_file.c_str());
                if (!ifs_param.is_open()) {
                    std::cout << "Can not Open parameter config file: " << config_parameters_file << "\n";
                    return false;
                } else {
                    store(parse_config_file(ifs_param, cmdline_options), config_variables_map);
                    notify(config_variables_map);
                }
            }

        }
        catch(po::error& e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << cmdline_options << std::endl;
            return false;
        }

        return true;
    }

    bool RequestObjectProposals(int frame, po::variables_map &options_map,
                                std::function<std::vector<GOT::segmentation::ObjectProposal>(po::variables_map &)> proposal_gen_fnc,
                                std::vector<GOT::segmentation::ObjectProposal> &proposals_out,
                                bool save_if_not_avalible=true) {

        // Check if proposals exist.
        // Yes -> load.
        // No -> run proposal generator.

        if (options_map.count("object_proposal_path")) {
            // There is path, see if files exist first, otherwise compute (and optionally store).

            char proposal_path_buff[MAX_PATH_LEN];
            snprintf(proposal_path_buff, MAX_PATH_LEN, options_map.at("object_proposal_path").as<std::string>().c_str(), frame);

            auto success_loading_proposals = GOT::segmentation::utils::LoadObjectProposals(proposal_path_buff, proposals_out);

            if (!success_loading_proposals) {
                printf("Could not load proposals, computing (note: processing will slow-down!) ...\r\n");
                proposals_out = proposal_gen_fnc(options_map);

                if (save_if_not_avalible) {
                    std::cout << "Saving proposals to: " << proposal_path_buff << std::endl;

                    boost::filesystem::path prop_path(proposal_path_buff);
                    boost::filesystem::path prop_dir = prop_path.parent_path();
                    SUN::utils::IO::MakeDir(prop_dir.c_str());

                    GOT::segmentation::utils::SaveObjectProposals(proposal_path_buff, proposals_out);
                }
            }

            return true;
        }

        return false;
    }
}

/*
  -------------
  Debug Levels:
  -------------
  0 - Outputs basically nothing, except relevant error messages.
  1 - Console output, logging.
  2 - Quantitative evaluation.
  3 - Most relevant visual results (per-frame, eg. segmentation, tracking results, ...).
  4 - Point clouds (per-frame), less relevant visual results.
  5 - Additional possibly relevant frame output (segmentation 3D data, integrated models, ...).
  >=6 - All possible/necessary debug stuff. Should make everything really really really slow.
  */
int main(const int argc, const char** argv) {
    std::cout << "Hello from CIWT!" << std::endl;

    // -------------------------------------------------------------------------------
    // +++ Command Args Parser +++
    // -------------------------------------------------------------------------------
    po::variables_map variables_map;
    if (!CIWTApp::ParseCommandArguments(argc, argv, variables_map)) {
        std::cout << "Failed at ParseCommandArguments ... " << std::endl;
        return -1;
    }
    printf("Tracking mode: %s.\r\n", CIWTApp::tracking_mode_str.c_str());
    printf("Run tracker: %s.\r\n", CIWTApp::run_tracker ? "YES" : "NO");

    // -------------------------------------------------------------------------------
    // +++ Globals +++
    // -------------------------------------------------------------------------------
    const int num_frames = CIWTApp::end_frame-CIWTApp::start_frame; // This many frames will be processed.

    // Makes sure the relevant values are correct.
    assert(num_frames>0);
    assert(CIWTApp::debug_level>=0);

    // Quantitative evaluation result storage
    std::vector<SUN::utils::KITTI::TrackingLabel> tracker_result_labels;

    // -------------------------------------------------------------------------------
    // +++ Create output dirs +++
    // -------------------------------------------------------------------------------
    // You can add more output sub-dir's!
    std::string output_dir_visual_results;
    std::string output_dir_tracking_data;
    if (CIWTApp::debug_level>=3) {
        output_dir_visual_results = CIWTApp::output_dir + "/visual_results";
        bool make_dir_success = SUN::utils::IO::MakeDir(output_dir_visual_results.c_str());
        assert(make_dir_success);

        output_dir_tracking_data = CIWTApp::output_dir + "/tracking_data";
        make_dir_success = SUN::utils::IO::MakeDir(output_dir_tracking_data.c_str());
        assert(make_dir_success);
    }

    // -------------------------------------------------------------------------------
    // +++ Data handling +++
    // -------------------------------------------------------------------------------
    SUN::utils::dirty::DatasetAssitantDirty dataset_assistant(variables_map); // Data loading hacky module
    PointCloudRGBA::Ptr left_point_cloud_preprocessed; // Preprocessed cloud
    std::vector<GOT::segmentation::ObjectProposal> object_proposals_all; // 'Raw' object proposals set
    std::shared_ptr<GOT::tracking::Resources> resource_manager(new GOT::tracking::Resources(variables_map["tracking_temporal_window_size"].as<int>()));

    // -------------------------------------------------------------------------------
    // +++ Visual odometry module +++
    // -------------------------------------------------------------------------------
    std::shared_ptr<libviso2::VisualOdometryStereo> vo_module = nullptr;

    auto InitVO = [](std::shared_ptr<libviso2::VisualOdometryStereo> &vo, double f, double c_u, double c_v, double baseline) {
        if (vo==nullptr) {
            libviso2::VisualOdometryStereo::parameters param;
            param.calib.f = f;
            param.calib.cu = c_u;
            param.calib.cv = c_v;
            param.base = baseline;
            vo.reset(new libviso2::VisualOdometryStereo(param));
        }
    };

    // -------------------------------------------------------------------------------
    // +++ Tracker +++
    // -------------------------------------------------------------------------------
    // Create the tracker object
    std::unique_ptr<GOT::tracking::ciwt_tracker::CIWTTracker> multi_object_tracker(new GOT::tracking::ciwt_tracker::CIWTTracker(variables_map));

    if (CIWTApp::debug_level>=3)
        multi_object_tracker->set_verbose(true);

    // -------------------------------------------------------------------------------
    // +++ Visualization Threads +++
    // -------------------------------------------------------------------------------

    /// 3D visualizer thread
    CIWTApp::visualization_3d_point_cloud.reset(new PointCloudRGBA);
    std::unique_ptr<boost::thread> visualization_3d_thread;
    if (CIWTApp::show_visualization_3d) {
        visualization_3d_thread.reset(new boost::thread(CIWTApp::VisualizeScene3D));
    }

    /// 2D visualization threads
    if (CIWTApp::show_visualization_2d) {
        cv::namedWindow("tracking_2d_window");
        cv::startWindowThread();

        cv::namedWindow("observations_id_window");
        cv::startWindowThread();
    }

    // -------------------------------------------------------------------------------
    // +++ MAIN TRACKING LOOP +++
    // -------------------------------------------------------------------------------
    double total_processing_time = 0.0;
    for (int current_frame=CIWTApp::start_frame; current_frame<=CIWTApp::end_frame; current_frame++) {

        if(CIWTApp::debug_level>0){
            std::cout << "\33[33;40;1m" << "-----------------------------------------------------------------------------" << "\33[0m" << std::endl;
            std::cout << "\33[33;40;1m" <<"                                  Processing frame " << current_frame << "\33[0m"<< std::endl;
            std::cout << "\33[33;40;1m" << "-----------------------------------------------------------------------------" << "\33[0m" << std::endl;
        }

        // -------------------------------------------------------------------------------
        // +++ Load data +++
        // -------------------------------------------------------------------------------

        if (!dataset_assistant.LoadData(current_frame, variables_map["dataset_name"].as<std::string>())) {
            printf("Dataset assistant can't load data :'( Check your config. \r\n");
            return -1;
        }

        auto &left_camera = dataset_assistant.left_camera_;
        auto &right_camera = dataset_assistant.right_camera_;
        const auto &left_image = dataset_assistant.left_image_;
        const auto &planar_ground_model = left_camera.ground_model();
        const auto &velocity_map = dataset_assistant.velocity_map_;
        left_point_cloud_preprocessed.reset(new PointCloudRGBA);
        pcl::copyPointCloud(*dataset_assistant.left_point_cloud_, *left_point_cloud_preprocessed);
        std::vector<SUN::utils::Detection> detections_current_frame = dataset_assistant.object_detections_;

        // -------------------------------------------------------------------------------
        // +++ Run visual odometry module => estimate egomotion +++
        // -------------------------------------------------------------------------------

        // Estimate ego
        InitVO(vo_module, left_camera.f_u(), left_camera.c_u(), left_camera.c_v(), dataset_assistant.stereo_baseline_); // Will be only initialized once, but need to do it within the loop
        Eigen::Matrix4d ego_this_frame = GOT::tracking::utils::EstimateEgomotion(*vo_module, dataset_assistant.left_image_, dataset_assistant.right_image_);

        // Accumulated transformation
        CIWTApp::egomotion = CIWTApp::egomotion * ego_this_frame.inverse();

        // Update left_camera, right_camera using estimated pose transform
        left_camera.ApplyPoseTransform(CIWTApp::egomotion);
        right_camera.ApplyPoseTransform(CIWTApp::egomotion);

        // -------------------------------------------------------------------------------
        // +++ 3D proposals +++
        // -------------------------------------------------------------------------------

        if(CIWTApp::debug_level>0) printf("->Processing object proposals ...\r\n");
        bool success_loading_proposals = false;
        if (CIWTApp::tracking_mode_str == "detection_shape")  { // In detection-only mode, don't bother with proposals
            success_loading_proposals = CIWTApp::RequestObjectProposals(current_frame, variables_map,
                                                                        std::bind(GOT::segmentation::proposal_generation::ComputeSuppressedMultiScale3DProposals,
                                                                                  dataset_assistant.left_point_cloud_, dataset_assistant.left_camera_, dataset_assistant.right_camera_, std::placeholders::_1),
                                                                        object_proposals_all);

            assert(success_loading_proposals);
        }

        // Filter certain 'noise' proposals (reject small ones and flying-ones)
        object_proposals_all = GOT::segmentation::utils::FilterProposals(object_proposals_all, left_camera, variables_map);

        // Sort (or: Re-Sort) proposals by their score.
        std::sort(object_proposals_all.begin(), object_proposals_all.end(),
                  [](const GOT::segmentation::ObjectProposal &i, const GOT::segmentation::ObjectProposal &j) { return i.score() > j.score(); });

        if(CIWTApp::debug_level>0) printf("->Got %d proposals.\r\n", static_cast<int>(object_proposals_all.size()));

        // Start per-frame timing analysis here!
        std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();

        // -------------------------------------------------------------------------------
        // +++ Observation fusion +++
        // -------------------------------------------------------------------------------
        if(CIWTApp::debug_level>0) printf("->Observation fusion ...\r\n");

        std::vector<GOT::tracking::Observation> observations_all;
        const auto &proposal_set_to_use = object_proposals_all;

        if (CIWTApp::tracking_mode_str=="detection") {
            /// Baseline: use only what can be inferred from detection bounding-boxes (faster, less accurate in 3D).
            auto det_obs_fnc = GOT::tracking::observation_processing::DetectionsOnly;
            observations_all = det_obs_fnc(detections_current_frame, proposal_set_to_use, left_image);
        }
        else if (CIWTApp::tracking_mode_str=="detection_shape") {
            /// Do proposal<->detection association using CRF, for precise detection localization.
            observations_all = GOT::tracking::observation_processing::DetectionToSegmentationFusion(
                    detections_current_frame, proposal_set_to_use,
                    left_camera, left_image, variables_map
            );
        }
        else {
            std::cout << "Unknown tracking_mode: " + CIWTApp::tracking_mode_str + "." << std::endl;
            return 0;
        }

        // -------------------------------------------------------------------------------
        // +++ Re-compute pos. cov. matrices +++
        // -------------------------------------------------------------------------------
        // Add some (learned) 'bias' to covariance matrices, depending on wheter it was localized with an 3D segment or by footpoint-projection.
        if(CIWTApp::debug_level>0) printf("->Override cov-mats ...\r\n");
        for (auto &obs:observations_all) {
            Eigen::Matrix3d cov_out;
            left_camera.ComputeMeasurementCovariance3d(obs.footpoint().head<3>(), 0.5, left_camera.P().block(0,0,3,4), right_camera.P().block(0,0,3,4), cov_out);

            double add_unc_x = 0.0;
            double add_unc_z = 0.0;
            if (obs.proposal_3d_avalible()) {
                add_unc_x = variables_map["pos_unc_seg_x"].as<double>();
                add_unc_z = variables_map["pos_unc_seg_z"].as<double>();
            } else {
                add_unc_x = variables_map["pos_unc_det_x"].as<double>();
                add_unc_z = variables_map["pos_unc_det_z"].as<double>();
            }

            cov_out(0, 0) += add_unc_x;
            cov_out(2, 2) += add_unc_z;
            obs.set_covariance3d(cov_out);
        }

         // -------------------------------------------------------------------------------
         // +++ Compute velocity for segments +++
         // -------------------------------------------------------------------------------
        if (dataset_assistant.velocity_map_.data!=nullptr) {

            // If velocity map is provided, we can compute velocities for fused observations
            if(CIWTApp::debug_level>0) printf("->Compute segment velocities ...\r\n");

            for (auto &obs:observations_all) {
                if (obs.proposal_3d_avalible()) {
                    Eigen::Vector3d obs_velocity = SUN::utils::observations::ComputeVelocity(velocity_map, obs.pointcloud_indices(),
                                                                                             variables_map["dt"].as<double>());
                    obs.set_velocity(obs_velocity);
                }
            }
        }

        const int max_num_observations_for_tracker = variables_map["max_num_input_observations"].as<int>();
        const int num_best_observations_to_consider = std::min(max_num_observations_for_tracker, static_cast<int>(observations_all.size()));
        auto observations_to_pass_to_tracker = observations_all;

        // -------------------------------------------------------------------------------
        // +++ Tracking +++
        // -------------------------------------------------------------------------------
        /// Feed the resource manager with current-frame data
        resource_manager->AddNewMeasurements(current_frame, left_point_cloud_preprocessed, left_camera);
        resource_manager->AddNewObservations(observations_to_pass_to_tracker);

        /// Invoke the tracker
        if (CIWTApp::run_tracker) {
            if(CIWTApp::debug_level>0) printf("->Running the tracker (observations: %d) ...\r\n", (int)observations_to_pass_to_tracker.size());
            multi_object_tracker->ProcessFrame(resource_manager, current_frame);
        }

        // Timing analysis
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        total_processing_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin).count();

        auto hypos_to_process_further = multi_object_tracker->selected_hypotheses(); // Selected hypothesis set.
        auto hypos_terminated = multi_object_tracker->terminated_hypotheses(); // Terminated, but still active, set.


        /// Export tracking data to labels struct (need for results export)
        GOT::tracking::utils::HypothesisSetToLabels(current_frame, hypos_to_process_further, tracker_result_labels,
                                                    std::bind(GOT::tracking::utils::HypoToLabelDefault, std::placeholders::_1, std::placeholders::_2,
                                                              left_camera.width(), left_camera.height()));

        // -------------------------------------------------------------------------------
        // +++ Visualizations (image) +++
        // -------------------------------------------------------------------------------

        /// Draw observations and tracked objects
        cv::Mat left_image_with_observations = left_image.clone();
        cv::Mat left_image_with_hypos_2d = left_image.clone();
        cv::Mat left_image_with_hypos_3d = left_image.clone();
        cv::Mat left_image_with_detections = left_image.clone();
        cv::Mat left_image_topdown;

        if (CIWTApp::debug_level>=2 || CIWTApp::show_visualization_2d) {
            CIWTApp::tracking_visualizer.DrawObservations(observations_all, left_image_with_observations, left_camera, GOT::tracking::draw_observations::DrawObservationAndOrientation);
            CIWTApp::tracking_visualizer.DrawHypotheses(hypos_to_process_further, left_camera, left_image_with_hypos_2d, GOT::tracking::draw_hypos::DrawHypothesis2d);
            CIWTApp::tracking_visualizer.DrawHypotheses(hypos_to_process_further, left_camera, left_image_with_hypos_3d, GOT::tracking::draw_hypos::DrawHypothesis3d);
            SUN::utils::visualization::DrawDetections(detections_current_frame, left_image_with_detections);
            CIWTApp::tracking_visualizer.DrawTrajectoriesBirdEye(hypos_to_process_further, left_point_cloud_preprocessed, left_camera, left_image_topdown);
        }

        if (CIWTApp::show_visualization_2d) {
            cv::imshow("tracking_2d_window", left_image_with_hypos_2d);
            cv::imshow("observations_id_window", left_image_with_observations);
        }

        /// Save to the disk
        if (CIWTApp::debug_level>=2) {
            char frame_str_buff[50];
            snprintf(frame_str_buff, 50, (CIWTApp::sequence_name + "_%06d").c_str(), current_frame);
            char output_path_buff[500];

            // Detections
            snprintf(output_path_buff, 500, "%s/detections_%s.png", output_dir_visual_results.c_str(), frame_str_buff);
            cv::imwrite(output_path_buff, left_image_with_detections);

            if (CIWTApp::run_tracker) {
                // Tracking 2D visualization
                snprintf(output_path_buff, 500, "%s/hypos_2d_%s.png", output_dir_visual_results.c_str(), frame_str_buff);
                cv::imwrite(output_path_buff, left_image_with_hypos_2d);

                snprintf(output_path_buff, 500, "%s/hypos_3d_%s.png", output_dir_visual_results.c_str(), frame_str_buff);
                cv::imwrite(output_path_buff, left_image_with_hypos_3d);

                // Tracking top-down visualization
                snprintf(output_path_buff, 500, "%s/hypos_topdown_%s.png", output_dir_visual_results.c_str(), frame_str_buff);
                cv::imwrite(output_path_buff, left_image_topdown*255);
            }
        }

        // -------------------------------------------------------------------------------
        // +++ Update the state for the 3D visualizer thread +++
        // -------------------------------------------------------------------------------

        if (CIWTApp::show_visualization_3d) {
            boost::mutex::scoped_lock updateLock(CIWTApp::visualization_3d_update_mutex);
            CIWTApp::visualization_3d_update_flag = true;

            pcl::copyPointCloud(*dataset_assistant.left_point_cloud_, *CIWTApp::visualization_3d_point_cloud);
            CIWTApp::visualization_3d_point_cloud->sensor_origin_ = Eigen::Vector4f(0.0,0.0,0.0,1.0);
            CIWTApp::visualization_observations = observations_all;
            CIWTApp::visualization_3d_proposals = proposal_set_to_use;

            char frame_str_buff[50];
            snprintf(frame_str_buff, 50, (CIWTApp::sequence_name).c_str());
            char buff[500];
            snprintf(buff, 500, "%s/3d_viewer_%s_%06d.png", output_dir_visual_results.c_str(), frame_str_buff, current_frame);
            CIWTApp::viewer_3D_output_path = std::string(buff);

            CIWTApp::visualization_3d_tracking_hypotheses = hypos_to_process_further;
            CIWTApp::visualization_3d_tracking_terminated_hypotheses = hypos_terminated;
            CIWTApp::visualization_3d_left_camera = left_camera;

            updateLock.unlock();
        }
    }

    // -------------------------------------------------------------------------------
    // +++ END-OF MAIN TRACKING LOOP +++
    // -------------------------------------------------------------------------------

    printf ("================ PROCESSING TIME ================\n");
    printf ("Total processing time %f (milisec) for %d frames.\n", total_processing_time, CIWTApp::end_frame-CIWTApp::start_frame);
    printf ("That is %f (milisec) per-frame.\n", total_processing_time / static_cast<double>(CIWTApp::end_frame-CIWTApp::start_frame));
    printf ("You like?\n");
    printf ("=================================================\n");

    // KITTI I/O label export obj.
    SUN::utils::KITTI::LabelsIO kitti_tracking_output_io;

    /// Write-out tracking quantitative results
    std::string tracking_output_dir = CIWTApp::output_dir + "/data";
    SUN::utils::IO::MakeDir(tracking_output_dir.c_str());
    char hypothesis_file_in_kitti_format[500];
    snprintf(hypothesis_file_in_kitti_format, 500, "%s/%s.txt", tracking_output_dir.c_str(), CIWTApp::sequence_name.c_str());
    kitti_tracking_output_io.WriteLabels(tracker_result_labels, hypothesis_file_in_kitti_format);

    std::cout << "Finished, yay!" << std::endl;
    return 0;
}
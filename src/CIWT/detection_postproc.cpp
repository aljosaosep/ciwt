//
// Created by Aljosa Osep on 14.05.20.
//

#include "detection_postproc.h"

namespace po = boost::program_options;

std::vector<SUN::utils::Detection> ProcDet2D(const SUN::utils::Camera &left_camera,
                                                const SUN::utils::Camera &right_camera,
                                                std::vector<SUN::utils::KITTI::TrackingLabel> dets,
                                                const po::variables_map &variables_map) {

    std::vector<SUN::utils::Detection> detections;
    for (const auto &det : dets) {
        SUN::utils::Detection detobj;
        Eigen::Vector4d bbox_2d;
        bbox_2d << det.boundingBox2D[0], det.boundingBox2D[1], det.boundingBox2D[2] - det.boundingBox2D[0],
                det.boundingBox2D[3] - det.boundingBox2D[1];
        detobj.set_bounding_box_2d(bbox_2d);
        detobj.set_score(det.score);
        detobj.set_category(static_cast<int>(det.type));
        detections.push_back(detobj);
    }


    /// Non-max-supp
    //detections = SUN::utils::detection::NonMaximaSuppression(detections, variables_map.at("detection_nms_iou").as<double>());

    /// Threshold
    auto f_filt = [](const SUN::utils::Detection &detection, const po::variables_map &variables_map)->bool {
        auto det_type = static_cast<SUN::shared_types::CategoryTypeKITTI> (detection.category());
        double threshold = 1e2;
        if (det_type==SUN::shared_types::CAR)
            threshold = variables_map["detection_threshold_car"].as<double>();
        else if (det_type==SUN::shared_types::PEDESTRIAN)
            threshold = variables_map["detection_threshold_pedestrian"].as<double>();
        else if (det_type==SUN::shared_types::CYCLIST)
            threshold = variables_map["detection_threshold_cyclist"].as<double>();
        return detection.score() >= threshold;
    };

    /// Filter by det. scores
    detections = SUN::utils::detection::ScoreFilter(detections, std::bind(f_filt, std::placeholders::_1, variables_map));

    /// SVM score -> quasi-probability (apply sigmoid ... but this is not really a probability!)
    // TODO softmax would make more sense
    std::for_each(detections.begin(), detections.end(), [](SUN::utils::Detection &det) {
        det.set_score(1.0 / (1.0 + std::exp(-1.0*det.score())));
    });

    /// Project to 3D
    detections = SUN::utils::detection::ProjectTo3D(detections, left_camera, right_camera);

    /// Geom. filter
    detections = SUN::utils::detection::GeometricFilter(detections, left_camera);

    return detections;
}



std::vector<SUN::utils::Detection> ProcDet3D(const SUN::utils::Camera &left_camera,
                                             const SUN::utils::Camera &right_camera,
                                             std::vector<SUN::utils::KITTI::TrackingLabel> dets,
                                             const po::variables_map &variables_map) {

    std::vector<SUN::utils::Detection> detections;

    // 3DOP detections were already preprocessed, just simply fill the det struct.
    for (const auto &det_label:dets) {
        SUN::utils::Detection det;

        Eigen::Vector4d pos = Eigen::Vector4d(det_label.location[0], det_label.location[1], det_label.location[2], 1.0);
        det.set_score(det_label.score);
        det.set_bounding_box_2d(Eigen::Vector4d(det_label.boundingBox2D[0], det_label.boundingBox2D[1],
                                                det_label.boundingBox2D[2] - det_label.boundingBox2D[0],
                                                det_label.boundingBox2D[3] - det_label.boundingBox2D[1]));

        det.set_category(SUN::shared_types::CAR); //det_label.type);
        det.set_footpoint(pos);

        // Compute pose covariance
        Eigen::Matrix3d detection_cov3d;

        SUN::utils::Camera::ComputeMeasurementCovariance3d(pos.head<3>(), 0.5,
               left_camera.P().block(0,0,3,4), right_camera.P().block(0,0,3,4),
                /*calib.GetProjCam2(), calib.GetProjCam3(),*/
                detection_cov3d);

        Eigen::Matrix3d gaussian_prior;
//                    gaussian_prior.setIdentity();
//                    gaussian_prior(0, 0) *= 0.2;
//                    gaussian_prior(2, 2) *= 0.2;
//                    detection_cov3d += gaussian_prior;
        det.set_pose_covariance_matrix(detection_cov3d);

        det.set_observation_angle(det_label.rotationY);

        Eigen::VectorXd bbox_3d;
        bbox_3d.setZero(6);
        bbox_3d << det_label.location[0], det_label.location[1], det_label.location[2],
                det_label.dimensions[1], det_label.dimensions[0], det_label.dimensions[2]; // Not sure if correct !!!

        det.set_bounding_box_3d(bbox_3d);
        detections.push_back(det);
    }

    return detections;
}

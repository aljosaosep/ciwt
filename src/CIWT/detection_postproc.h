//
// Created by Aljosa Osep on 14.05.20.
//

#ifndef CIWT_DETECTION_POSTPROC_H
#define CIWT_DETECTION_POSTPROC_H

// Boost
#include <boost/program_options.hpp>

// Why fwd dcl. no work here?!?
#include "sun_utils/camera.h"
#include "detection.h"
#include "utils_kitti.h"



std::vector<SUN::utils::Detection> ProcDet2D(const SUN::utils::Camera &left_camera,
                                             const SUN::utils::Camera &right_camera,
                                             std::vector<SUN::utils::KITTI::TrackingLabel> dets,
                                             const boost::program_options::variables_map &variables_map);



std::vector<SUN::utils::Detection> ProcDet3D(const SUN::utils::Camera &left_camera,
                                             const SUN::utils::Camera &right_camera,
                                             std::vector<SUN::utils::KITTI::TrackingLabel> dets,
                                             const boost::program_options::variables_map &variables_map);

#endif //CIWT_DETECTION_POSTPROC_H

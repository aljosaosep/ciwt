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

#ifndef GOT_DETECTIONS_H
#define GOT_DETECTIONS_H

// Std
#include <deque>

// PCL
#include <pcl/common/common_headers.h>

// Tracking
#include <tracking/observation.h>

// OpenCV
#include <opencv2/core/core.hpp>

// Forward declarations
namespace SUN { namespace utils { class Camera; } }


namespace GOT {
    namespace tracking {
        typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;
        typedef std::deque<PointCloudRGBA::Ptr> PointCloudQueue;
        typedef std::deque<SUN::utils::Camera> CameraQueue;
        typedef std::deque<std::vector<Observation> > ObservationQueue;

        /**
           * @brief Implements Resources class. The task of this class is to hold the past measurements in a certain temporal window.
           *        Measurements that fall outside of the temporal window (specified by temporal_window_size) are dumped.
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class Resources {
        public:
            Resources(int temporal_window_size);

            /**
               * @brief With this method, we add most important new-frame measurements and increase the frame count.
               */
            void AddNewMeasurements(int frame, PointCloudRGBA::ConstPtr reference_cloud, const SUN::utils::Camera &camera);

            // Getters / setters
            SUN::utils::Camera GetCamera(int frame, bool &lookup_success) const;
            bool GetCamera(int frame, SUN::utils::Camera &cam) const;
            PointCloudRGBA::ConstPtr GetPointCloud(int frame) const;

            void AddNewObservations(const std::vector<Observation> &observations);
            std::vector<Observation> GetObservations(int frame, bool &lookup_success) const;
            bool GetInlierObservation(int frame, int inlier_index, Observation &obs) const;

            // Typedefs
            typedef std::shared_ptr<const Resources> ConstPtr;

        protected:
            /**
                * @brief This method maps global frame number to queue query index.
                * @return 'true' if we can get out valid index, 'false' otherwise. (ie. data, with specified query frame number was already dropped).
            */
            bool FrameIndexToQueueIndex(const int frame, int &queue_index) const;

            PointCloudQueue scene_cloud_queue_; // Point clouds queue.
            CameraQueue camera_queue_; // Camera maps queue.
            ObservationQueue observation_queue_;
            int temporal_window_size_; // Defines how many past measurements we store.
            int current_frame_; // Internal frame counter.
        };
    }
}

#endif

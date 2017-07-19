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

// tracking lib
#include <tracking/resource_manager.h>

// pcl
#include <pcl/io/io.h>

// utils
#include "sun_utils/camera.h"

namespace GOT {
    namespace tracking {
        Resources::Resources(int temporal_window_size) {
            temporal_window_size_ = temporal_window_size;
            current_frame_ = 0;
        }

        bool Resources::FrameIndexToQueueIndex(const int frame, int &queue_index) const {
            // queue_index = this->temporal_window_size_ - (current_frame_-frame)-1; // Maps frame id to queue lookup id
            queue_index = this->scene_cloud_queue_.size() - (current_frame_ - frame) - 1;
            if (queue_index >= 0 && queue_index < scene_cloud_queue_.size())
                return true;
            return false;
        }

        SUN::utils::Camera Resources::GetCamera(int frame, bool &lookup_success) const {
            int queue_lookup_index = -1;
            lookup_success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);
            if (lookup_success)
                return camera_queue_.at(queue_lookup_index);
            // else
            // std::cout << "Resources::GetCamera::Error: index out of queue bounds! " << std::endl;
            return SUN::utils::Camera();
        }

        bool Resources::GetCamera(int frame, SUN::utils::Camera &cam) const {
            int queue_lookup_index = -1;
            bool lookup_success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);
            if (lookup_success) {
                cam = camera_queue_.at(queue_lookup_index);
                return true;
            }

            return false;
        }

        PointCloudRGBA::ConstPtr Resources::GetPointCloud(int frame) const {
            int queue_lookup_index = -1;
            bool lookup_success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);

            if (lookup_success)
                return this->scene_cloud_queue_.at(queue_lookup_index);

            return nullptr;
        }

        void Resources::AddNewMeasurements(int frame, PointCloudRGBA::ConstPtr reference_cloud,
                                           const SUN::utils::Camera &camera) {
            // Make a new copy of the point cloud
            PointCloudRGBA::Ptr cloud_copy(new PointCloudRGBA);
            pcl::copyPointCloud(*reference_cloud, *cloud_copy);

            if (scene_cloud_queue_.size() >=
                static_cast<unsigned int>(this->temporal_window_size_)) { // Make sure queue size is never > T!
                // Point Cloud
                scene_cloud_queue_.pop_front();
                scene_cloud_queue_.push_back(cloud_copy);
                camera_queue_.pop_front();
                camera_queue_.push_back(camera);
            }
            else {
                scene_cloud_queue_.push_back(cloud_copy);
                camera_queue_.push_back(camera);
            }

            current_frame_ = frame;
        }

        void Resources::AddNewObservations(const std::vector<Observation> &observations) {
            if (this->observation_queue_.size() >= static_cast<unsigned int>(this->temporal_window_size_)) {
                observation_queue_.pop_front();
                observation_queue_.push_back(observations);
            }
            else {
                observation_queue_.push_back(observations);
            }
        }


        std::vector<Observation> Resources::GetObservations(int frame, bool &lookup_success) const {
            int queue_lookup_index = -1;
            lookup_success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);

            if (lookup_success)
                return observation_queue_.at(queue_lookup_index);
            // TODO (Aljosa) enable back!
            //else
            //    std::cout << "Resources::GetObservations::Error: index out of bounds! " << std::endl;
            return std::vector<Observation>(); // Return empty, notify the guy via lookup_success flag
        }

        bool Resources::GetInlierObservation(int frame, int inlier_index, Observation &obs) const {
            bool ret_val = false;
            int queue_lookup_index = -1;
            bool lookup_success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);

            if (lookup_success) {
                const auto &obs_vec = observation_queue_.at(queue_lookup_index);
                if (obs_vec.size()>inlier_index) {
                    obs = obs_vec.at(inlier_index);
                    ret_val = true;
                }
            }

            return ret_val;
        }
    }
}

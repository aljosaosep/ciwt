/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Francis Engelmann, Aljosa Osep (engelmann, osep -at- vision.rwth-aachen.de)

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

#ifndef SUN_DISPARITY_H
#define SUN_DISPARITY_H

// std
#include <string>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace SUN {

    class DisparityMap
    {
    public:
        DisparityMap() {};
        DisparityMap(const SUN::DisparityMap &map);
        DisparityMap(const std::string file_name);
        DisparityMap(const cv::Mat disp);
        DisparityMap(const unsigned int width, const unsigned int height);
        DisparityMap(const float *data, const unsigned int width, const unsigned int height);

        unsigned int width() const { return (uint)disparity_map_.cols; }
        unsigned int height() const { return (uint)disparity_map_.rows; }
        cv::Mat mat() const { return disparity_map_; }

        void Read(const std::string file_name, unsigned int scaling = 256);
        void ReadDaimler(const std::string file_name, unsigned int scaling = 512);
        void Write(const std::string file_name, unsigned int scaling = 256) const;

        float GetDisp(const int32_t v, const int32_t u) const { return disparity_map_.at<float>(v,u); }
        void SetDisp(const int32_t v, const int32_t u, const float val) { disparity_map_.at<float>(v,u) = val; }

        bool IsValid(const int32_t v, const int32_t u) const { return disparity_map_.at<float>(v,u) >= 0; }
        bool SetInvalid(const int32_t v, const int32_t u) { disparity_map_.at<float>(v,u) = -1; }

        void ReadDisparityMap(const std::string file_name, unsigned int scaling = 256);
        void ReadDisparityMapDaimler(const std::string file_name, unsigned int scaling = 512);
        void WriteDisparityMap(const std::string file_name, unsigned int scaling = 256) const;

    private:
        cv::Mat disparity_map_;
    };

}

#endif // SUN_DISPARITY_H

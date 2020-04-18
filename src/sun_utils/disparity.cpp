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

#include "disparity.h"

#include <iostream>

namespace SUN {

    DisparityMap::DisparityMap(const SUN::DisparityMap &disparityMap) {
        this->disparity_map_ = disparityMap.mat().clone();
    }

    DisparityMap::DisparityMap(const std::string file_name) {
        this->Read(file_name);
    }

    DisparityMap::DisparityMap(const cv::Mat disp) {
        if(disp.depth() != CV_32F) {
            throw std::runtime_error("The cv::Mat to init a SUN::DisparityMap should have depth CV_32F");
        }
        disparity_map_ = disp.clone();
    }

    DisparityMap::DisparityMap(const unsigned int width, const unsigned int height) {
        disparity_map_ = cv::Mat(height, width, CV_32F, cv::Scalar(0));
    }

    DisparityMap::DisparityMap(const float *data, const unsigned int width, const unsigned int height) {
        disparity_map_ = cv::Mat(height, width, CV_32F);
        memcpy(disparity_map_.data, data, width*height*sizeof(float));
    }

    void DisparityMap::Read(const std::string file_name, unsigned int scaling) {
        disparity_map_.data = 0;
        ReadDisparityMap(file_name, scaling);
    }

    void DisparityMap::ReadDaimler(const std::string file_name, unsigned int scaling) {
        disparity_map_.data = 0;
        ReadDisparityMapDaimler(file_name, scaling);
    }

    void DisparityMap::Write(const std::string file_name, unsigned int scaling) const {
        WriteDisparityMap(file_name, scaling);
    }

    void DisparityMap::ReadDisparityMap(const std::string file_name, unsigned int scaling) {
        cv::Mat image16 = cv::imread(file_name,cv::IMREAD_ANYDEPTH);
        disparity_map_ = cv::Mat(image16.rows, image16.cols, CV_32F);

        // v = y
        // u = x
        for (int32_t v=0; v<disparity_map_.rows; v++) {
            for (int32_t u=0; u<disparity_map_.cols; u++) {
                uint16_t val = image16.at<unsigned short>(v,u);
                if (val==0) SetInvalid(v, u);
                else SetDisp(v, u, ((float) val) / scaling);
            }
        }
    }

    void DisparityMap::ReadDisparityMapDaimler(const std::string file_name, unsigned int scaling) {
        cv::Mat image16 = cv::imread(file_name, cv::IMREAD_ANYDEPTH);
        disparity_map_ = cv::Mat(image16.rows, image16.cols, CV_32F);

        for (int32_t v=0; v<disparity_map_.rows; v++) {
            for (int32_t u=0; u<disparity_map_.cols; u++) {
                uint16_t val = image16.at<unsigned short>(v,u);
                if (val == 65535 || val < 0)
                    SetInvalid(v, u);
                else
                    SetDisp(v, u, (float) val / scaling);
            }
        }
    }

    void DisparityMap::WriteDisparityMap(const std::string file_name, unsigned int scaling) const {
        cv::Mat image16 = cv::Mat(disparity_map_.rows, disparity_map_.cols, CV_16U);
        for (int32_t v=0; v<disparity_map_.rows; v++) {
            for (int32_t u=0; u<disparity_map_.cols; u++) {
                if (IsValid(v, u))
                    image16.at<unsigned short>(v,u) = (uint16_t)(std::max((double)(GetDisp(v, u)*scaling),1.0));
                else
                    image16.at<unsigned short>(v,u) = 0;
            }
        }
        cv::imwrite(file_name, image16);
    }
}

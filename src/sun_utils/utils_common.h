/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Francis Engelmann (osep, Engelmann -at- vision.rwth-aachen.de)

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

#ifndef GOT_UTILS_COMMON_H
#define GOT_UTILS_COMMON_H

// std
#include <vector>

// eigen
#include <Eigen/Core>

namespace SUN {
    namespace utils {

        /**
         * @brief Converts a flat index into a tuple of coordinate arrays
        */
        void UnravelIndex(int index, int width, int *x, int *y);

        /**
         * @brief Return 'flattened' index
         */
        void RavelIndex(int x, int y, int width, int *index);
    }
}
#endif //GOT_UTILS_COMMON_H

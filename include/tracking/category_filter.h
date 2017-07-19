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

#ifndef GOT_CATEGORY_FILTER_H
#define GOT_CATEGORY_FILTER_H

#include "category_filter.h"

// std
#include <vector>
#include <functional>

namespace GOT {
    namespace tracking {
        namespace bayes_filter {
            int GetArgMax(const std::vector<double> &posterior);
            std::vector<double> CategoryFilter(const std::vector<double> &likelihood, const std::vector<double> &prior);
        }
    }
}
#endif //GOT_CATEGORY_FILTER_H

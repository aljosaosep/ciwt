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

#ifndef GOT_POTENTIAL_FUNCTIONS_H
#define GOT_POTENTIAL_FUNCTIONS_H

namespace GOT { namespace tracking {  class Hypothesis; } }
namespace GOT { namespace tracking {  class HypothesisInlier; } }

// std
#include <vector>
#include <functional>

// boost
#include <boost/program_options.hpp>

// CIWT
#include "observation_fusion.h"

namespace po = boost::program_options;

namespace GOT {
    namespace tracking {
        namespace CRF {

            typedef std::function<double(const GOT::tracking::Hypothesis &, const GOT::tracking::Hypothesis &, int)> OverlapFnc;

            namespace CIWT {
                namespace hypothesis_selection {

                    namespace overlap {
                        double OverlapFncRectangleIoU(const GOT::tracking::Hypothesis &hypothesis_1, const GOT::tracking::Hypothesis &hypothesis_2, int frame);
                    }

                    namespace potential_func {
                        double HypoUnary(const Hypothesis &hypo, int current_frame, const po::variables_map &parameter_map);
                        double HypoPairwise(const Hypothesis &hypo_1, const Hypothesis &hypo_2, int current_frame, const po::variables_map &parameter_map);
                    }

                    std::vector<GOT::tracking::HypothesisInlier> IntersectInliersDefault(
                            const GOT::tracking::Hypothesis &hypothesis_1,
                            const GOT::tracking::Hypothesis &hypothesis_2,
                            int frame);


                    double ComputePairwiseOverlap(const GOT::tracking::Hypothesis &hypothesis_1, const GOT::tracking::Hypothesis &hypothesis_2,
                                                  int frame, OverlapFnc overlap_fnc);


                }

                namespace observation_fusion {
                    double UnaryFuncLinearCombination(const std::vector<double> &potentials, const po::variables_map &parameter_maps);
                }
            }
        }
    }
}

#endif //GOT_POTENTIAL_FUNCTIONS_H

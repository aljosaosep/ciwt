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

#include "tracking/category_filter.h"

// std
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>

namespace GOT {
    namespace tracking {

        namespace bayes_filter {

            std::vector<double> CategoryFilter(const std::vector<double> &likelihood,
                                               const std::vector<double> &prior) {
                assert(likelihood.size()==prior.size());

                // Compute normalization fact.
                double norm_fact = 0.0;
                for (int i=0; i<likelihood.size(); i++) {
                    norm_fact += likelihood.at(i)*prior.at(i);
                }

                assert (norm_fact > 0.01);

                // Compute the posterior
                std::vector<double> posterior(likelihood.size());

                for (int i=0; i<likelihood.size(); i++) {
                    posterior.at(i) = likelihood.at(i)*prior.at(i) / norm_fact;
                }

                // If likelihood is uniform, apply some exponential decay
                const double uniform_val = 1.0 / static_cast<double>(likelihood.size());
                if (std::fabs(likelihood.at(0)-uniform_val) < 0.01) {
                    // Assuming that sum(likelihood)==1, checking-out only first element is a legit thing to do.
                    // In case this does not hold, you're screwed up anyway!
                    const double decay_rate = 1.0;
                    for (auto &val:posterior) {
                        const double exp_decay = std::exp(decay_rate*(0.5-val));
                        val *= exp_decay;
                    }
                }

                // Let's make sure we don't get degenerate distrib.
                bool degenerate = false;
                for (const auto &val:posterior) {
                    if (val > 0.99)
                        degenerate = true;
                }

                // Apply exp. decay if prob. distrib. is becoming degenerate.
                if (degenerate) {
                    for (auto &val:posterior) {
                        if (val>0.99)
                            val = 0.99;
                        else
                            val = 0.1 / static_cast<double>(posterior.size());
                    }
                }

                // Unit test: see that the posterior sums-up to 1
                double sum = std::accumulate(posterior.begin(), posterior.end(), 0.0);

                assert(sum < 1.1);

                if (sum>1.1) {
                    printf("Bayes Filter Panic: sum(posterior) > 1.1! Fix urgently needed!\r\n");
                }

                return posterior;
            }

            int GetArgMax(const std::vector<double> &posterior) {
                return static_cast<int>(std::distance(posterior.begin(), std::max_element(posterior.begin(), posterior.end())));
            }
        }
    }
}
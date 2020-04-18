/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Dennis Mitzel (osep, mitzel -at- vision.rwth-aachen.de)

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

#ifndef GOT_QPBO_H
#define GOT_QPBO_H

#include <Eigen/Core>

#include <vector>

namespace GOT {
    namespace tracking {

        namespace QPBO {
            void FindMax(std::vector<std::pair<double, int> > &models,
                         int start, double score,
                         int *ex_steps, const Eigen::MatrixXd &Q,//double *ex_a,
                         std::vector<int>& ex_bestind, double &ex_bestscore, int ex_bestsize, int ex_dim,
                         int STEP_SIZE);

            /**
               * @brief Solves QPBO probem:  m'Qm -> max, s.t. m \in {0, 1}. Greedy approach, yields approximate solution.
               * @note Implementation by Dennis Mitzel.
               * @param Q The cost matrix (quadratic, symmetric), containing hypo. potentials on the diagonal, hypo. iteractions on off-diagonal.
               * @param m Indicator vector. Indicates, wheter the hypo was selected or not (1... selected, 0... not selected).
               * @return Best score.
               */
            double SolveGreedy(const Eigen::MatrixXd &Q, Eigen::VectorXi &m);

            /**
               * @brief Solves QPBO probem:  m'Qm -> max, s.t. m \in {0, 1}. Exact approach. Suitable for small matrices Q.
               * @note Implementation by Dennis Mitzel. Implements multi-branch method by Schindler etal.
               * @param Q The const matrix (quadratic, symmetric), containing hypo. potentials on the diagonal, hypo. iteractions on off-diagonal.
               * @param m Indicator vector. Indicates, wheter the hypo was selected or not (1... selected, 0... not selected).
               * @return Best score.
               */
            double SolveMultiBranch(const Eigen::MatrixXd &Q, Eigen::VectorXi &m);
        }
    }
}

#endif

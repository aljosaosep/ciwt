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

// tracking lib
#include <tracking/qpbo.h>
#include <iostream>

namespace GOT {
    namespace tracking {
        namespace QPBO {

            double SolveGreedy(const Eigen::MatrixXd &Q, Eigen::VectorXi &m) {
                assert(Q.rows()==Q.cols()); // Make sure it's a square matrix!
                const int matrix_dim = Q.rows();

                // Init indicator vector m
                m.resize(matrix_dim);
                m.setZero();

                std::vector<double> result(matrix_dim);
                std::vector<int> b_dropped(matrix_dim);
                std::fill(b_dropped.begin(), b_dropped.end(), 0);

                double best_score = 0.0;
                double opt_score = 0.0;
                int opt_index = 0;
                double old_score = 0.0;

                Eigen::VectorXi copy_m;

                for (int i=0; i<matrix_dim; i++) {
                    copy_m = m;
                    for (int j=0; j<matrix_dim; j++) {
                        if (m[j] > 0 || b_dropped.at(j) > 0) continue;

                        copy_m[j] = 1;
                        double eval_current_state = copy_m.cast<double>().transpose()*Q*copy_m.cast<double>();

                        result.at(j) = eval_current_state;

                        if (result.at(j)<0)
                            b_dropped.at(j) = 1;
                        if (m[j] != 1)
                            copy_m[j] = 0;
                    }

                    auto result_max_it = std::max_element(result.begin(), result.end());
                    opt_score = *result_max_it;
                    opt_index = std::distance(result.begin(), result_max_it);
                    std::fill(result.begin(), result.end(), 0.0);

                    if(old_score < opt_score) {
                        m[opt_index] = 1;
                        old_score = opt_score;
                        best_score = opt_score;
                    }
                    else {
                        break;
                    }
                }

                return best_score;
            }

            double SolveMultiBranch(const Eigen::MatrixXd &Q, Eigen::VectorXi &m) {
                assert(Q.rows()==Q.cols()); // Make sure it's a square matrix!
                const  int ex_dim = Q.rows();

                // Init indicator vector m
                m.resize(ex_dim);
                m.setZero();

                const int step_size = 6; // TODO (Aljosa) why 6?
                int ex_steps[step_size];
                std::vector<int> ex_best_indices;
                ex_best_indices.clear();
                double ex_best_score = 0.0;
                int ex_best_size = 0;

                // Set number of potential search branches to a tracable value
                ex_steps[2] = 5;
                ex_steps[3] = 2;
                ex_steps[4] = ex_steps[5] = 1;
                if (ex_dim < 50) { // 50000 paths max
                    ex_steps[0] = ex_steps[1] = 50;
                }
                else if (ex_dim < 70) { // 25000 paths max
                    ex_steps[0] = 70;
                    ex_steps[1] = 25;
                }
                else if (ex_dim < 100) { // 10000 paths max
                    ex_steps[0] = 50;
                    ex_steps[1] = 20;
                }
                else if (ex_dim < 250) { // 5000 paths max
                    ex_steps[0] = 50;
                    ex_steps[1] = 10;
                }
                else { // 2500 paths max
                    ex_steps[0] = 50;
                    ex_steps[1] = 5;
                }

                std::vector<std::pair<double, int> > models(ex_dim);
                for (int i=0; i<ex_dim; i++)
                    models[i].second = i;


                FindMax(models, 0, 0.0, ex_steps, Q, ex_best_indices, ex_best_score, ex_best_size, ex_dim, step_size);

                for (int i=0; i<ex_best_indices.size(); i++) {
                    m[ex_best_indices.at(i)] = 1;
                    // Here, one could also read-out, or return, the corresponding hypo scores!
                }

                return ex_best_score;
            }

            void FindMax(std::vector<std::pair<double, int> > &models,
                         int start, double score,
                         int *ex_steps, const Eigen::MatrixXd &Q,
                         std::vector<int>& ex_best_ind, double &ex_best_score, int ex_best_size, int ex_dim,
                         int step_size) {

                if (score > ex_best_score) {
                    ex_best_score = score;
                    ex_best_ind.resize(start);
                    for (int i=0; i<start; i++)
                        ex_best_ind[i] = models[i].second;
                    ex_best_size = ex_best_ind.size();
                }

                if (start >= ex_dim)
                    return;

                // Compute the effect of this model
                for (int i=start; i<ex_dim; i++) {
                    int m_idx = models[i].second;
                    double inc = 0;
                    for (int j=0; j<start; j++)
                        inc += Q(m_idx, models[j].second); // Aljosa: I think this is correct order!
                    inc = 2.0*inc + Q(m_idx, m_idx);
                    models[i].first = inc;
                }

                // Sort the remaining models according to their merit
                if (start < (ex_dim-1)) {
                    std::vector<std::pair<double, int> > models_copy = models;
                    std::sort(models_copy.begin()+start, models_copy.end(), std::greater< std::pair<double, int> >());
                    models.clear();

                    for (unsigned i=0; i<models_copy.size(); i++) {
                        models.push_back(models_copy.at(i));
                    }
                }

                // Try selecting the remaining models
                int step_no = 1;
                if (start < step_size)
                    step_no = ex_steps[start];
                if (start + step_no > ex_dim)
                    step_no = ex_dim - start;
                for (int i=start; i<(start+step_no); i++) {
                    if (models[i].first > 0) {
                        // Follow this branch recursively
                        double inc = models[i].first;
                        int idx = models[i].second;
                        std::swap(models[start], models[i]);
                        std::vector<std::pair<double, int> > models_x = models;
                        FindMax(models_x, start+1, inc+score, ex_steps, Q, ex_best_ind, ex_best_score, ex_best_size, ex_dim, step_size);

                        for (int j=(start+1); j<ex_dim; j++) {
                            if (models[j].second == idx) {
                                std::swap(models[start], models[j]);
                                break;
                            }
                        }
                    }
                    else
                        break;
                }
            }
        }
    }
}

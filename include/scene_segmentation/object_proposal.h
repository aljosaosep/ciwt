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

#ifndef GOT_OBJECT_PROPOSAL_H
#define GOT_OBJECT_PROPOSAL_H

// PCL
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>

// Boost
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>

namespace boost { namespace serialization {

        template<   class Archive,
                class S,
                int Rows_,
                int Cols_,
                int Ops_,
                int MaxRows_,
                int MaxCols_>
        inline void serialize(Archive & ar,
                              Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & matrix,
                              const unsigned int version)
        {
            int rows = matrix.rows();
            int cols = matrix.cols();
            ar & make_nvp("rows", rows);
            ar & make_nvp("cols", cols);
            matrix.resize(rows, cols); // no-op if size does not change!

            // always save/load row-major
            for(int r = 0; r < rows; ++r)
                for(int c = 0; c < cols; ++c)
                    ar & make_nvp("val", matrix(r,c));
        }

        template<   class Archive,
                class S,
                int Dim_,
                int Mode_,
                int Options_>
        inline void serialize(Archive & ar,
                              Eigen::Transform<S, Dim_, Mode_, Options_> & transform,
                              const unsigned int version)
        {
            serialize(ar, transform.matrix(), version);
        }
    }
} // namespace boost::serialization

namespace GOT {
    namespace segmentation {

        /**
           * @brief This class represents one object proposal. Object proposal is 3D region, mostly defined by set of 3d points. These 3d points are stored here as
           *        (linear) indices, corresponding to 'cells' of (organized) pcl::PointCloud. Thus, indices relate the region to both, 2d image plane and a set of 3d points.
           *        2d indices may can be obtained using GOT::utils::geometry::ConvertIndex2Coordinates(...)
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class ObjectProposal {

            //! For serialization.
            friend class boost::serialization::access;
            template<class Archive>
            void serialize(Archive & ar, const unsigned int version) {
                ar & groundplane_indices_;
                ar & pointcloud_indices_;
                ar & score_;
                ar & pos3d_;
                ar & bounding_box_2d_;
                ar & bounding_box_3d_;
                ar & bounding_box_2d_ground_plane_;
                ar & pose_covariance_matrix_;
                ar & scale_pairs_;
                ar & convex_hull_;
            }

        public:

            void init(const std::vector<int> &groundplane_indices, const std::vector<int> &pointcloud_indices, const Eigen::Vector4d &bounding_box_2d, const Eigen::VectorXd &bounding_box_3d,  int proposal_id, double score);

            // Setters / getters
            Eigen::Vector4d pos3d() const;
            double score() const;
            const Eigen::Vector4d& bounding_box_2d() const;
            const Eigen::VectorXd &bounding_box_3d() const;
            const Eigen::Vector4d& bounding_box_2d_ground_plane() const;

            const std::vector<int>& pointcloud_indices() const;
            const std::vector<int>& ground_plane_indices() const;

            const std::vector<std::pair<float, float> >& scale_pairs() const;
            const Eigen::Matrix3d& pose_covariance_matrix() const;

            void set_bounding_box_2d(const Eigen::Vector4d &bbox2d);
            void set_bounding_box_3d(const Eigen::VectorXd &bbox3d);
            void set_bounding_box_2d_ground_plane(const Eigen::Vector4d &bbox2d);
            void set_pos3d(const Eigen::Vector4d &pos3d);
            void set_score(double new_score);
            void set_pointcloud_indices(const std::vector<int> &indices);
            void set_groundplane_indices(const std::vector<int> &indices);
            void set_pose_covariance_matrix(const Eigen::Matrix3d& pose_cov_mat);

            void add_scale_pair(const std::pair<float, float> &scale_pair);
            void add_scale_pairs(const std::vector<std::pair<float, float> > &scale_pairs);

            void set_convex_hull(const std::vector<std::pair<int, int>> &convex_hull);
            const std::vector<std::pair<int, int>>& convex_hull() const;

        private:
            std::vector<int> pointcloud_indices_;
            std::vector<int> groundplane_indices_;

            Eigen::Vector4d pos3d_;
            double score_;

            Eigen::Vector4d bounding_box_2d_; // 2D bounding box: [min_x min_y width height]
            Eigen::VectorXd bounding_box_3d_; // 3D bounding box: [centroid_x centroid_y centroid_z width height length q.w q.x q.y q.z]
            Eigen::Vector4d bounding_box_2d_ground_plane_;

            Eigen::Matrix3d pose_covariance_matrix_;

            // Scales info
            std::vector<std::pair<float, float> > scale_pairs_;
            std::vector<std::pair<int, int>> convex_hull_;
        };
    }
}

#endif

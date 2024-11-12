#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.h"
#include "visualization_msgs/msg/marker_array.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>

#include <thread>

#include "dbscan/dbscan.h"

#define TIMING

class LidarDBSCANNode : public rclcpp::Node {
public:
    LidarDBSCANNode() : Node("dbscan_node") {
        // ROS2 퍼블리셔 및 서브스크라이버 설정
        pub_clustered_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dbscan/cluster_cloud", 1);
        pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/dbscan/cluster_centroids", 1);
        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/carla/ego_vehicle/lidar", 1, std::bind(&LidarDBSCANNode::cloudCallback, this, std::placeholders::_1));
    }

private:
    /**
     * DBSCAN clusatering Thread
     * @param cloud Input point cloud
     * @param eps Searching radius
     * @param minPts Minimum number of the points in searching radius
     */
    void runDbscanThread(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float eps, int minPts) {
        DBSCAN dbscan(cloud, eps, minPts);
        dbscan.runDbscan();
    }

    /**
     * Create boundary of cluster using visualization_msgs::msg::MarkerArray
     * @param cluster
     * @param clusterId
     * @param header
     * @param markerArray
     */
    void createClusterBoundaryMarkers(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster, int clusterId,
                                      const std_msgs::msg::Header& header, visualization_msgs::msg::MarkerArray& markerArray) {
        pcl::ConvexHull<pcl::PointXYZI> convexHull;
        pcl::PointCloud<pcl::PointXYZI>::Ptr hullPoints(new pcl::PointCloud<pcl::PointXYZI>());
        convexHull.setInputCloud(cluster);
        convexHull.reconstruct(*hullPoints);

        visualization_msgs::msg::Marker lineStrip;
        lineStrip.header = header;
        lineStrip.ns = "cluster_boundaries";
        lineStrip.id = clusterId;
        lineStrip.type = visualization_msgs::msg::Marker::CUBE_LIST;//LINE_STRIP;
        lineStrip.action = visualization_msgs::msg::Marker::ADD;
        lineStrip.pose.orientation.w = 1.0;
        lineStrip.scale.x = 0.1;
        lineStrip.scale.y = 0.1;
        lineStrip.scale.z = 0.1;

        lineStrip.color.r = static_cast<int>(clusterId * 50) % 255 / 255.0;
        lineStrip.color.g = static_cast<int>(clusterId * 80) % 255 / 255.0;
        lineStrip.color.b = static_cast<int>(clusterId * 120) % 255 / 255.0;
        lineStrip.color.a = 1.0;

        for (const auto& point : hullPoints->points) {
            geometry_msgs::msg::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            lineStrip.points.push_back(p);
        }

        if (!hullPoints->points.empty()) {
            geometry_msgs::msg::Point p;
            p.x = hullPoints->points.front().x;
            p.y = hullPoints->points.front().y;
            p.z = hullPoints->points.front().z;
            lineStrip.points.push_back(p);
        }

        markerArray.markers.push_back(lineStrip);
    }

    /**
     * Filtering the point cloud (PassThrough Filter)
     * @param inputCloud Input point cloud before filtering
     * @param filteredCloud Output point cloud (Filtering the inputCloud)
     */
    void filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& inputCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filteredCloud) {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(inputCloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-15.0, 15.0);
        pass.filter(*filteredCloud);

        pass.setInputCloud(filteredCloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-15.0, 15.0);
        pass.filter(*filteredCloud);

        pass.setInputCloud(filteredCloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-2.0, 2.0);
        pass.filter(*filteredCloud);
    }

    // 클라우드 콜백 함수
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloudMsg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cCloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloudMsg, *cCloud);

        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cCloud);
        vg.setLeafSize(0.1f, 0.1f, 0.1f);
        pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>());
        vg.filter(*filteredCloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr rangeFilteredCloud(new pcl::PointCloud<pcl::PointXYZ>());
        filterPointCloud(filteredCloud, rangeFilteredCloud);

        // if (rangeFilteredCloud->empty()) {
        //     RCLCPP_WARN(this->get_logger(), "Filtered cloud is empty after PassThrough.");
        //     return;
        // }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (const auto& point : rangeFilteredCloud->points) {
            pcl::PointXYZI cvtCloud;
            cvtCloud.x = point.x;
            cvtCloud.y = point.y;
            cvtCloud.z = point.z;
            cvtCloud.intensity = UNCLASSIFIED;
            cloud->points.push_back(cvtCloud);
        }

        float eps = 1;
        int minPts = 6;

        #ifdef TIMING // Debugging for computation time
            auto tic = this->now();
        #endif

        std::thread dbscanThread(&LidarDBSCANNode::runDbscanThread, this, cloud, eps, minPts);
        dbscanThread.join();

        #ifdef TIMING // Debugging for computation time
            auto toc = this->now();
            RCLCPP_INFO(this->get_logger(), "DBSCAN took %f seconds", (toc - tic).seconds());
        #endif

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud, output);
        output.header = cloudMsg->header;
        pub_clustered_cloud_->publish(output);

        visualization_msgs::msg::MarkerArray markerArray;
        int clusterId = 0;
        std::map<float, std::vector<pcl::PointXYZI>> clusters;
        for (const auto& point : cloud->points) {
            if (point.intensity != NOISE) {
                clusters[point.intensity].push_back(point);
            }
        }

        for (const auto& cluster : clusters) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZI>());
            clusterCloud->points.insert(clusterCloud->points.end(), cluster.second.begin(), cluster.second.end());
            createClusterBoundaryMarkers(clusterCloud, clusterId++, cloudMsg->header, markerArray);
        }

        pub_markers_->publish(markerArray);
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_clustered_cloud_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarDBSCANNode>());
    rclcpp::shutdown();
    return 0;
}
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>

#include <thread>
#include <map>
#include <optional>
#include <cmath>

#include "dbscan/dbscan.h"  // 사용자가 구현한 DBSCAN 라이브러리 헤더 파일

#define TIMING
#ifdef TIMING
    #include <chrono>
#endif

class DetectionNode : public rclcpp::Node {
public:
    DetectionNode() : Node("radar_detection_node") {
        // ROS2 퍼블리셔 및 서브스크라이버 설정
        pub_clustered_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dbscan/radar/clusters", 1);
        pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/dbscan/radar/clusters/bboxes", 1);
        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/carla/ego_vehicle/radar_front", 1, std::bind(&DetectionNode::cloudCallback, this, std::placeholders::_1));
    }

private:
    void runDbscanThread(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float eps, int minPts) {
        DBSCAN dbscan(cloud, eps, minPts);
        dbscan.runDbscan();
    }

    std::optional<visualization_msgs::msg::Marker> createBoundingBox(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster, int id, const std_msgs::msg::Header& header) {

        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);

        float z_center = (min_pt.z() + max_pt.z()) / 2.0;
        float width = max_pt.x() - min_pt.x();
        float length = max_pt.y() - min_pt.y();
        float height = max_pt.z() - min_pt.z();

        if (height < 1.2 || height > 4.0 ||
            width < 0.05 || width > 6.5 ||
            length < 0.15 || length > 4.0) return std::nullopt;

        visualization_msgs::msg::Marker bbox;
        bbox.header = header;
        bbox.ns = "radar_bounding_" + std::to_string(id);
        bbox.id = id;
        bbox.type = visualization_msgs::msg::Marker::CUBE;
        bbox.action = visualization_msgs::msg::Marker::ADD;
        bbox.pose.position.x = (min_pt.x() + max_pt.x()) / 2.0;
        bbox.pose.position.y = (min_pt.y() + max_pt.y()) / 2.0;
        bbox.pose.position.z = z_center;
        bbox.scale.x = width;
        bbox.scale.y = length;
        bbox.scale.z = height;
        bbox.color.r = 1.0f;
        bbox.color.g = 0.0f;
        bbox.color.b = 0.0f;
        bbox.color.a = 0.5;
        bbox.text = "Radar_Obstacle_" + std::to_string(id);
        return bbox;
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloudMsg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cCloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloudMsg, *cCloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (const auto& point : cCloud->points) {
            pcl::PointXYZI cvtCloud;
            cvtCloud.x = point.x;
            cvtCloud.y = point.y;
            cvtCloud.z = point.z;
            cvtCloud.intensity = UNCLASSIFIED;
            cloud->points.push_back(cvtCloud);
        }

        float eps = 0.5;
        int minPts = 6;

        #ifdef TIMING
            auto tic = this->now();
        #endif

        std::thread dbscanThread(&DetectionNode::runDbscanThread, this, cloud, eps, minPts);
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

            // visualization_msgs::msg::Marker bbox = createBoundingBox(clusterCloud, clusterId++, cloudMsg->header);
            auto bbox = createBoundingBox(clusterCloud, clusterId++, cloudMsg->header);
            if (bbox) {
                float volume = bbox->scale.x * bbox->scale.y * bbox->scale.z;
                float density = static_cast<float> (clusterCloud->points.size()) / volume;

                if (density > 0.5) {
                    markerArray.markers.push_back(*bbox);
                }
            }
        }

        pub_markers_->publish(markerArray);
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_clustered_cloud_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DetectionNode>());
    rclcpp::shutdown();
    return 0;
}

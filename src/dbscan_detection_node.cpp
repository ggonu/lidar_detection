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
    DetectionNode() : Node("detection_node") {
        // ROS2 퍼블리셔 및 서브스크라이버 설정
        pub_clustered_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dbscan/clusters", 1);
        pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/dbscan/clusters/bboxes", 1);
        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/carla/ego_vehicle/semantic_lidar", 1, std::bind(&DetectionNode::cloudCallback, this, std::placeholders::_1));
    }

private:
    // DBSCAN 클러스터링 스레드
    void runDbscanThread(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float eps, int minPts) {
        DBSCAN dbscan(cloud, eps, minPts);
        dbscan.runDbscan();
    }

    // 클러스터로부터 3D Bounding Box 생성
    std::optional<visualization_msgs::msg::Marker> createBoundingBox(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster, int id, const std_msgs::msg::Header& header) {

        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);

        // Bounding Box 중심의 z 값과 크기 조건 검사
        float z_center = (min_pt.z() + max_pt.z()) / 2.0;
        float width = max_pt.x() - min_pt.x();
        float length = max_pt.y() - min_pt.y();
        float height = max_pt.z() - min_pt.z();

        // 차량 크기 범위 조건 설정
        // if (z_center < -0.5 || z_center > 0.5 || width < 0.2 || width > 3.0 || length < 0.2 || length > 5.0 || height < 0.4 || height > 3.5) {
        //     return std::nullopt;  // 크기 또는 높이가 조건을 벗어나면 제외
        // }

        if (height < 1.2 || height > 4.0 ||
            width < 0.05 || width > 6.5 ||
            length < 0.15 || length > 4.0) return std::nullopt;

        visualization_msgs::msg::Marker bbox;
        bbox.header = header;
        bbox.ns = "dbscan_bounding_boxes";
        bbox.id = id;
        bbox.type = visualization_msgs::msg::Marker::CUBE;
        bbox.action = visualization_msgs::msg::Marker::ADD;
        bbox.pose.position.x = (min_pt.x() + max_pt.x()) / 2.0;
        bbox.pose.position.y = (min_pt.y() + max_pt.y()) / 2.0;
        bbox.pose.position.z = z_center;
        bbox.scale.x = width;
        bbox.scale.y = length;
        bbox.scale.z = height;
        bbox.color.r = 0.0f;
        bbox.color.g = 1.0f;
        bbox.color.b = 0.0f;
        bbox.color.a = 0.5;  // 투명도 설정
        return bbox;
    }

    // 포인트 클라우드 필터링
    void filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& inputCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filteredCloud) {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(inputCloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-10.0, 10.0);
        pass.filter(*filteredCloud);

        pass.setInputCloud(filteredCloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-5.0, 50.0);
        pass.filter(*filteredCloud);

        pass.setInputCloud(filteredCloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-2.3, 2.7);
        pass.filter(*filteredCloud);
    }

    // 콜백 함수: LiDAR 데이터를 받아서 클러스터링 수행 및 3D BBox 생성
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloudMsg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cCloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloudMsg, *cCloud);

        // 거리 및 각도 필터링 적용
        pcl::PointCloud<pcl::PointXYZ>::Ptr rangeFilteredCloud(new pcl::PointCloud<pcl::PointXYZ>());
        filterPointCloud(cCloud, rangeFilteredCloud);

        // 다운샘플링
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(rangeFilteredCloud);
        vg.setLeafSize(0.5f, 0.5f, 0.5f);
        pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>());
        vg.filter(*filteredCloud);

        if (filteredCloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Filtered cloud is empty after PassThrough.");
            return;
        }

        // 클러스터링을 위해 포인트를 XYZI 형식으로 변환
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (const auto& point : filteredCloud->points) {
            pcl::PointXYZI cvtCloud;
            cvtCloud.x = point.x;
            cvtCloud.y = point.y;
            cvtCloud.z = point.z;
            cvtCloud.intensity = UNCLASSIFIED;
            cloud->points.push_back(cvtCloud);
        }

        // DBSCAN 파라미터
        float eps = 1.0;
        int minPts = 8;

        #ifdef TIMING
            auto tic = this->now();
        #endif

        // DBSCAN 클러스터링 실행
        std::thread dbscanThread(&DetectionNode::runDbscanThread, this, cloud, eps, minPts);
        dbscanThread.join();

        #ifdef TIMING // Debugging for computation time
            auto toc = this->now();
            RCLCPP_INFO(this->get_logger(), "DBSCAN took %f seconds", (toc - tic).seconds());
        #endif

        // 클러스터링된 포인트 클라우드 퍼블리시
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud, output);
        output.header = cloudMsg->header;
        pub_clustered_cloud_->publish(output);

        // 각 클러스터의 Bounding Box 생성 및 퍼블리시
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

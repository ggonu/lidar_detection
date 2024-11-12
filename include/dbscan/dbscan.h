#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>
#include <cstddef>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>


#define BORDER 2.0
#define CORE 1.0
#define UNCLASSIFIED -1.0
#define NOISE -2.0


/*      pcl::PointXYZI        *
 *                            *
 * float x, y, z, intensity   *
 *                            *
 * intensity is for clusterId */


class DBSCAN {
private:
    pcl::PointCloud<pcl::PointXYZI>::Ptr mCloud; // SetOfPoints
    size_t mPtSize;
    unsigned int mMinPts;   // MinPts
    float mEps;         // Eps
    


public:

    DBSCAN(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float eps, int minPts)
        : mCloud(cloud), mPtSize(cloud->size()), mEps(eps), mMinPts(minPts){
        // mPt = minPts;
        // mEps = eps;
        // mCloud = cloud;
    }
    ~DBSCAN(){}

    int runDbscan();
    bool expandCluster(int idx, float clusterId);

    std::vector<int> regionQuery(int idx);

    size_t getCloudSize() { return mCloud->size(); }
    int getMinPt() { return mMinPts; }
    float getEps() { return mEps; }

};


#endif  // DBSCAN_H
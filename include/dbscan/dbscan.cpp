#include "dbscan.h"

int DBSCAN::runDbscan() {
    float clusterId = 0.0;

    for (size_t i = 0; i < mCloud->points.size(); ++i) {
        if (mCloud->points[i].intensity == UNCLASSIFIED) {
            if (expandCluster(i, ++clusterId)) {}
            else mCloud->points[i].intensity = NOISE;
        }
    }

    return static_cast<int>(clusterId);
}

bool DBSCAN::expandCluster(int idx, float clusterId) {
    std::vector<int> seeds = regionQuery(idx);

    if (seeds.size() < mMinPts) {
        mCloud->points[idx].intensity = NOISE;
        return false;
    }

    for (int seedIdx : seeds) {
        mCloud->points[seedIdx].intensity = clusterId;
    }

    size_t currentIdx = 0;
    while (currentIdx < seeds.size()) {
        int currentPoint = seeds[currentIdx];
        std::vector<int> result = regionQuery(currentPoint);

        if (result.size() >= mMinPts) {
            for (int resultIdx : result) {
                if (mCloud->points[resultIdx].intensity == UNCLASSIFIED || mCloud->points[resultIdx].intensity == NOISE) {
                    if (mCloud->points[resultIdx].intensity == UNCLASSIFIED) {
                        seeds.push_back(resultIdx);
                    }
                    mCloud->points[resultIdx].intensity = clusterId;
                }
            }
        }
        currentIdx++;
    }

    return true;
}

std::vector<int> DBSCAN::regionQuery(int idx) {
    std::vector<int> neighbors;
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(mCloud);

    std::vector<float> pointDistance;
    std::vector<int> pointIdxSearch;

    if (kdtree.radiusSearch(mCloud->points[idx], mEps, pointIdxSearch, pointDistance) > 0) {
        neighbors.insert(neighbors.end(), pointIdxSearch.begin(), pointIdxSearch.end());
    }

    return neighbors;
}
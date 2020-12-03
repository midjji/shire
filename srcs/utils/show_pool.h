#pragma once
#include<opencv2/core.hpp>
#include <trackerinterface/h/featurepool.h>
namespace cvl{
cv::Mat3b draw_feature_pool(CFeaturePool& pool,
                            cv::Mat3b rgb);

cv::Mat3b draw_feature_pool_prediction(CFeaturePool& pool,
                                       cv::Mat3b rgb);
} // end namespace cvl

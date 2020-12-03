#pragma once
#include <opencv2/core.hpp>
#include "trackerinterface/h/featurepool.h"
#include <sfm/measurement.h>
namespace cvl{

cv::Mat3b draw_feature_pool(CFeaturePool pool, cv::Mat3b rgb, cv::Mat1f dim);
cv::Mat3f make_flow_image(CFeaturePool pool, cv::Mat1f dim);
cv::Mat3f make_flow_image(std::vector<std::shared_ptr<Measurement>> ms, cv::Mat1f dim);
cv::Mat3f subsample_flow(cv::Mat3f flow, int factor);
void safe_save_image(cv::Mat im, std::string path);
void make_flow_dataset(CFeaturePool pool,std::vector<std::shared_ptr<Measurement>> ms, cv::Mat1f dim,std::string basepath, int frameid);
cv::Mat3b draw_world_tracks(const std::vector<std::shared_ptr<Measurement>>& ms, cv::Mat3b rgb, double threshold);
cv::Mat3b draw_candidates(const std::vector<std::shared_ptr<Measurement>>& ms, cv::Mat3b rgb, double threshold);
}

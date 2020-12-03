#pragma once
#include <vector>
#include <sfm/measurement.h>
#include <mlib/vis/flow_viewer.h>
namespace cvl{
std::vector<Flow> get_imo_curr_mean_flow(std::vector<std::shared_ptr<Measurement>> ms);
std::vector<Flow> get_imo_mean_flow(std::vector<std::shared_ptr<Measurement>> ms, bool all=true);
PointCloud get_point_cloud(std::vector<std::shared_ptr<Measurement>> ms);
std::shared_ptr<FlowField> get_imo_mean_flow_field(std::shared_ptr<TimePoint> tp);
std::shared_ptr<FlowField> get_imo_flow_field(std::shared_ptr<TimePoint> tp);
}

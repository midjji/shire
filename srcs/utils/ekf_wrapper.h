
#pragma once
#include <memory>

#include <opencv2/highgui.hpp>
#include "paramHandling/h/paramseteditor_Qt4.h"
#include <6dVisionFilters/h/6dVisionFilters.h>
#include <6dVisionFilters/h/filterParameters.h>

#include <daimler/dataset.h>

#include <mlib/sfm/p3p/pnp_ransac.h>

#include <mlib/vis/flow_viewer.h>

#include "klt_cuda/h/klt_tracker.h"
#include "paramHandling/h/paramHandlingGUI.h"


namespace cvl{



class EkfWrapper
{
public:




    ParamSet* getParamSet();
    void init();
    void update(PoseD Pcp, CFeaturePool pool, cv::Mat1f dim);
    // delayed one step to match the vo systems estimate.
    std::shared_ptr<FlowField> get_flowfield();
    std::vector<Vector7d> get_filter_state();
    SixDVisionFilters::C6dVisionFiltersParameters filterparams;
    SixDVisionFilters::C6dVisionFilters filters;
    std::shared_ptr<FlowField> previous=nullptr;
    std::shared_ptr<FlowField> current=nullptr;
    std::vector<Vector7d> last_state; // x,y,z, vx,vy,vz,{0 => missing, 1=> good}
    bool inited=false;

};

}


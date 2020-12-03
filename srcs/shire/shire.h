#pragma once


#include <deque>

#include <mlib/utils/mlibtime.h>


#include "klt_cuda/h/klt_tracker.h"
#include "paramHandling/h/paramHandlingGUI.h"


#include <sfm/sample.h>
#include <map.h>
#include <imo.h>
#include <mlib/sfm/p3p/pnp_ransac.h>

namespace cvl{



class Shire{
public:

    Shire();


    // the hir one relies on the stream beeing synched and takes care of this itself! it relies on the circ buffer containing atleast 2 though...
    void operator()(std::shared_ptr<HirSample>& sd);
    ParamSet* getParamSet();
    void init(int rows=1024,
              int cols= 2048);
    void set_min_creation_count(int count);



    bool draw_raw_tracks=true;
    bool draw_new_imo=true;
    bool draw_candidates_b=true;
    bool draw_world_tracks_b=true;
    bool draw_anything_at_all=true;
    bool run_ekf=false;
    bool start_paused=true;
    bool make_flow_dataset_b=false;
    bool first=true;


    bool save_images=false;
    bool save_optical_flow=false;
    bool show_flow_b=false;
    int  flow_index=0;
    bool retrack_world=true;
    bool retrack_world_redo_pnp=true;
    bool pnp_only=false;
    bool use_labels=false;

    bool save_result=false;
    bool save_verified_depths=false;



    int display_index=0;
    int draw_increment=1;

    double reprojection_threshold=3;
    double disambiguation_threshold=4;
    double candidate_threshold=5;






private:
    ParamSet params;

    std::shared_ptr<KLT_CUDA_Tracker::CKLT_Tracker<unsigned short>> tracker=nullptr;
    std::shared_ptr<KLT_CUDA_Tracker::CKLT_Tracker<unsigned short>> retracker=nullptr;

    std::shared_ptr<Map> map;
    mlib::NamedTimerPack timers;

    bool inited=false;
    uint min_creation_count=18;
    uint min_inlier_count();


    // models
    std::shared_ptr<PoseImo> hir(std::vector<std::shared_ptr<Measurement>>& ms);

    void bundle_egomotion();
    void save_results(int result_nr);

    void display(std::shared_ptr<HirSample> sd,
                 std::shared_ptr<PoseImo> new_imo);
    void printTimers();
    void track(std::shared_ptr<HirSample>& sd);






    // frameid, imo id, imov, ekfv

    std::vector<Vdata> vdatas;



    void save_image(cv::Mat3b im, std::string name, int frameid);



    // includes the start time of the program
    std::string get_output_directory();
    PnpParams get_pnp_params();

    int prev_framestamp=-1;

    CFeaturePool raw_pool; // a copy of the raw tracking pool, for display


};
typedef std::shared_ptr<Shire> sShire;



















}

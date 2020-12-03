#pragma once
#include <mlib/utils/colormap.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/cvl/pose.h>
#include <sfm/feature.h>
#include <sfm/timepoint.h>



namespace cvl{
class ImoMassCenters{
public:
    std::vector<Vector3d> xms;
    mlib::Color color;
    ImoMassCenters( std::vector<Vector3d> a, mlib::Color b):xms(a),color(b){}
};
class Map{
public:
    Map();
    static std::shared_ptr<Map> create(uint size);
    std::shared_ptr<TimePoint> createTimePoint( uint frameid);


    Map(uint size=50000);

    void push(std::shared_ptr<TimePoint>& tp);
    std::vector<PoseD> getTrajectory();
    std::vector<std::shared_ptr<TimePoint>> get_last_n_timepoints(int n);
    std::vector<std::shared_ptr<TimePoint>> getTimePoints();

    std::shared_ptr<TimePoint> at(uint i);
    std::shared_ptr<TimePoint> previous(uint i=0);
    std::shared_ptr<TimePoint> at_frameid(uint frameid){
        auto it=tps_by_frameid.find(frameid);
        if(it==tps_by_frameid.end()) return nullptr;
        return it->second;
    }

    uint getSize();
    double velocity();


    std::vector<std::shared_ptr<PoseImo> > getImos(bool last_frame_only=true); //

    std::weak_ptr<Map> self;

    //frameid, pose
    std::map<int, PoseD> get_poses();
    // imoid, frameid, pose    
    std::map<std::shared_ptr<PoseImo>, std::map<int, PoseD>> get_imo_poses(bool in_world_coordinates_at_current_time=true);
    void add_imo(std::shared_ptr<PoseImo>& imo);
    /**
     * @brief get_imo_trajectories_in_world
     * @param prepend_egomotion
     * @return
     *
     * note that this gives you the camera trajectory in world.
     * The object trajectory in world is tricky...
     * We could talk about mass center trajectories though.
     *
     */
    std::vector<std::vector<PoseD>> get_imo_trajectories_in_world();
    // expensive!
    std::vector<ImoMassCenters> get_imo_mass_centers_in_world();
    // using that we can
    std::vector<std::vector<PoseD>> get_imo_object_coordinates_in_world();

    auto get_all_imos(){return imos;}

private:

    std::vector<std::shared_ptr<TimePoint>> tps;// the images have been removed from these!
    std::vector<std::shared_ptr<PoseImo>> imos;
    std::map<int,std::shared_ptr<TimePoint>> tps_by_frameid;
    int size;

};



}

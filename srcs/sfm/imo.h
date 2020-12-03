#pragma once
#include <sfm/map.h>
#include <mlib/utils/bounding_box.h>
#include <daimler/results_types.h>
namespace cvl{

double getReprojectionThreshold(double motiondisparity, double disparity, int age);
class IMO{
public:
    IMO(){}
    virtual ~IMO(){}

    static int count;
    uint time0;

    std::shared_ptr<PoseD> getPose(uint time) const{

        auto it= ps.find(time);
        if(it==ps.end()){
            return nullptr;
        }
        return it->second;
    }
    std::map<int,PoseD> get_poses(){
        std::map<int,PoseD> rs;
        for(auto [key,val]:ps){
            if(val==nullptr) {std::cout<<"nullptr pose in imo"<<std::endl;continue;}
            rs[key]=*val;
        }
        return rs;
    }
    int get_frameid_zero(){
        for(auto [key, val]:ps)
            return key;
        assert(false && "found no poses when looking for origin!");
        return -1;
    }

    std::vector<int> get_frameids(){
        std::vector<int> frameids;frameids.reserve(ps.size());
        for(auto [key, val]:ps)
            frameids.push_back(key);
        return frameids;
    }
    std::vector<int> get_last_n_frameids(int N){

        std::vector<int> ret; ret.reserve(N);
        std::vector<int> frameids=get_frameids();
        int n=0;
        for(auto it=frameids.rbegin();it!=frameids.rend() && n++<N;++it)
            ret.push_back(*it);
        return ret;
    }


    // gets the max reprojection error of the feature given the imo


    std::shared_ptr<PoseD> origin;
    int age();
    PoseD delta_pose();
    void set_pose(uint time, std::shared_ptr<PoseD> p){ps[time]=p;}
    bool has_pose(uint time){return ps.find(time)!=ps.end();}
    void remove_pose(uint time){ps.erase(time);}


private:
    std::map<uint,std::shared_ptr<PoseD>> ps;

};
std::vector<std::shared_ptr<Measurement>> not_nullptr_and_has_previous_filter(                  std::vector<std::shared_ptr<Measurement>>& ms );
std::vector<std::shared_ptr<Measurement>> not_nullptr_and_has_previous_and_is_inlier_filter(    std::vector<std::shared_ptr<Measurement>>& ms, double max_err );
std::vector<std::shared_ptr<Measurement>> not_nullptr_and_has_previous_and_not_imo_filter(      std::vector<std::shared_ptr<Measurement>>& ms );
std::vector<std::shared_ptr<Measurement>> not_nullptr_and_has_previous_and_imo_filter(          std::vector<std::shared_ptr<Measurement>>& ms);
std::vector<std::shared_ptr<Measurement>> not_nullptr_and_has_previous_and_is_car(              std::vector<std::shared_ptr<Measurement>>& ms);
std::map<std::shared_ptr<PoseImo>, std::vector<std::shared_ptr<Measurement>>> sort_by_imo(      std::vector<std::shared_ptr<Measurement>>& ms);



class PoseImo:public IMO
{
public:
    PoseImo();
    static std::shared_ptr<PoseImo>
    create(std::vector<std::shared_ptr<Measurement>>& ms, uint min_creation_count);

    ~PoseImo();


    BoundingBox get_imo_bounding_box(int frame_id, bool conservative) const;
    // performs pnp, then refine
    bool estimate(std::vector<std::shared_ptr<Measurement>>& ms,
                  int min_inlier_count,
                  double reprojection_threshold);
    // performs ba
    void refine(double reprojection_threshold);

    // note updates feature to have this imo and its best X
    double getMinReprojectionDistance(std::shared_ptr<Feature>& f) const;
    Vector3d find_initial_X_for(std::shared_ptr<Feature>& f) const;
    int id=-1;

    Vector3d median_centroid();
    Vector3d get_mass_center();

    void add_feature(std::shared_ptr<Feature>& f);
    void remove_feature(std::shared_ptr<Feature>& f);
    //frame id, pose


    bool valid(){return features.size()>3 && age()>1;}
    const std::set<std::shared_ptr<Feature>>& get_features(){return features;}

    imores get_imo_res();
    // very expensive, avoid if possible
    std::vector<std::shared_ptr<Measurement>> get_measurements_in_frame(int frame_id) const;

    mlib::Color get_color(){
        return mlib::Color::nrlarge(2+id % 16);
    }
    std::vector<std::shared_ptr<Feature>> get_recent_features(int oldest_frameid);

private:
    std::set<std::shared_ptr<Feature>> features;
    mlib::Timer batimer,problemtimer;
    std::weak_ptr<PoseImo> self;
    std::shared_ptr<Map> map=nullptr;
    std::map<int, BoundingBox> bbs;
};

Vector3d median_centroidvs(std::vector<Vector3d>& vs);
Vector3d median_centroidvs(std::vector<std::shared_ptr<Measurement>>& vs);
}

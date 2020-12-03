#include <map.h>
#include <mlib/utils/cvl/triangulate.h>

#include <iostream>
#include <ceres/rotation.h>
#include <imo.h>
#include <mlib/utils/string_helpers.h>
#include <opencv2/highgui.hpp>
#include <measurement.h>
#include <mlib/utils/simulator_helpers.h>
#include <mlib/utils/mlog/log.h>

#include <sfm/calibration.h>
using std::cout;using std::endl;
// imo

namespace cvl{




Map::Map(){}
std::shared_ptr<Map> Map::create(uint size){
    auto mp=std::make_shared<Map>(size);
    mp->self=mp;
    return mp;
}


Map::Map(uint size):size(size){
    tps.reserve(size);
}

void Map::push(std::shared_ptr<TimePoint>& tp){

    for(int i=0;i<int(tps.size()) - size;++i){
        tps[i]->ms.clear();
    }
    tps.push_back(tp);
}
std::vector<cvl::PoseD> Map::getTrajectory()
{


    PoseD P=HirCalibration::common().P_cam0_vehicle().inverse();
    std::vector<cvl::PoseD> ps;ps.reserve( tps.size());
    for(auto& tp:tps) ps.push_back(P*(*tp->Pnw));
    return ps;
}
std::vector<std::shared_ptr<TimePoint>> Map::getTimePoints(){
    return tps;
}
std::vector<std::shared_ptr<TimePoint>> Map::get_last_n_timepoints(int N){
    std::vector<std::shared_ptr<TimePoint>> rets,rev; rets.reserve(N);
    int n=0;
    for(auto it=tps.rbegin();it!=tps.rend() && n++<N;++it)
        rets.push_back(*it);
   for(auto it=rets.rbegin();it!=rets.rend();++it)
       rev.push_back(*it);
   return rev;
}


std::shared_ptr<TimePoint> Map::at(uint i){return tps.at(i);}
std::shared_ptr<TimePoint> Map::previous(uint i){return tps.at(tps.size()-i-1);}

uint Map::getSize(){return tps.size();}
double Map::velocity(){
    if(tps.size()<2) return 0;
    PoseD curr=*(tps.back()->Pnw);

    PoseD prev=*(previous(1)->Pnw);

    PoseD Pdelta = curr*prev.inverse();
    double delta_time=1.0/30.0;
    return Pdelta.getTinW().norm()/delta_time;

}

std::shared_ptr<TimePoint> Map::createTimePoint(uint frameid){
    auto tmp=self.lock();
    assert(tmp!=nullptr);
    auto tp= TimePoint::create(frameid,tmp);
    tps_by_frameid[frameid]=tp;
    return tp;

}

std::vector<std::shared_ptr<PoseImo> > Map::getImos(bool last_frame_only){
    if(!last_frame_only)
        return imos;
    std::vector<std::shared_ptr<PoseImo>> imosr;

    if(tps.size()<2) return imosr;
    std::set<std::shared_ptr<PoseImo>> imos;
    if(last_frame_only){
        for(auto& m:tps.back()->ms)
            if(m)
                if(m->feature->get_imo()!=nullptr)
                    imos.insert(m->feature->get_imo());
        imosr.reserve(imos.size());

        for(auto& imo:imos)
            if(imo){
                if(imo->getPose(tps.back()->frameid)==nullptr)
                    std::cout<<"get imos : imo seen in the current? frame failed to pnp. WTF?"<<endl;//exit(1);
                else
                    imosr.push_back(imo);
            }
        return imosr;
    }

    for(auto& tp:tps)
        for(auto& m:tp->ms)
            if(m){
                auto i=m->feature->get_imo();
                if(i)
                    imos.insert(i);
            }
    for(auto& imo:imos)
        imosr.push_back(imo);
    return imosr;
}
std::map<int, PoseD> Map::get_poses()
{
    std::map<int, PoseD> poses;
    for(auto& tp:tps) poses[tp->frameid]=*tp->Pnw;
    return poses;
}
// imoid, frameid, pose
std::map<std::shared_ptr<PoseImo>, std::map<int, PoseD>> Map::get_imo_poses(bool in_world){
    std::map<std::shared_ptr<PoseImo>, std::map<int, PoseD>> map;
    // frameid, pose
    auto egom=get_poses();

    for(auto& imo: imos){ // they are not in world...
        auto tmp=imo->get_poses();
        if(in_world){
            int first=imo->get_frameid_zero();
            assert(egom.find(first)!=egom.end());
            if(egom.find(first)==egom.end()) {mlog()<<"something really bad\n";exit(1);}
            PoseD Piw=egom[first]; // transform from world to imo zero.
            for(auto&[key, val]:tmp)
            {
                val=val*Piw;
            }
        }
        map[imo]=tmp;
    }
    return map;
}

template<class Key, class Element> std::vector<Element> elems2vector(const std::map<Key,Element>& map){
    std::vector<Element> vs;vs.reserve(map.size());
    for(auto [k,v]:map)
        vs.push_back(v);
    return vs;
}
std::vector<std::vector<PoseD>> Map::get_imo_trajectories_in_world(){
    std::vector<std::vector<PoseD>> vs;
    vs.push_back(getTrajectory());
    for(auto [key, map]:get_imo_poses(true)){
        vs.push_back(elems2vector(map));
    }
    return vs;
}

std::vector<std::vector<PoseD>> Map::get_imo_object_coordinates_in_world(){
    cout<<"not implemented!"<<endl;
    return std::vector<std::vector<PoseD>>();
}

std::vector<ImoMassCenters> Map::get_imo_mass_centers_in_world()
{
std::vector<ImoMassCenters> imomcs;imomcs.reserve(imos.size());


    auto Pnw=get_poses();
    for(auto& imo:imos)
    {

        if(imo->get_poses().size()<3) continue;// TODO replace with final size < 3

        auto ps=imo->get_poses();

        Vector3d xm=imo->get_mass_center();

        std::vector<Vector3d> xs; xs.reserve(1000);// in world
        for(auto [frameid, pose]:ps){

            assert(Pnw.find(frameid)!=Pnw.end());
            auto x=Pnw[frameid].inverse()*pose*xm;

            xs.push_back(x);
        }
        auto tmp=ImoMassCenters(xs,imo->get_color());
        imomcs.push_back(tmp);
    }
    return imomcs;
}
void Map::add_imo(std::shared_ptr<PoseImo>& imo){
    if(!imo->valid())
        mlog()<<"soemthing ery weird"<<endl;
    imos.reserve(10000);
    imos.push_back(imo);
}

}

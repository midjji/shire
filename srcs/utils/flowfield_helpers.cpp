#include <mlib/utils/mlog/log.h>
#include <mlib/utils/colormap.h>


#include <sfm/imo.h>

#include <utils/flowfield_helpers.h>


namespace cvl {
Vector3d toV3(mlib::Color color){
    return Vector3d(color[2],color[1],color[0])/255.0;
}
std::map<uint, Vector3d>
get_imo_feature_in_world_map(std::shared_ptr<Feature>& f){
    //cout<<"get_imo_feature_in_world"<<endl;
    std::vector<std::shared_ptr<Measurement>> ms = f->getMeasurements();

    // for each measurement get the imo and timepoint poses
    std::shared_ptr<PoseImo> imo=f->get_imo();
    if(imo==nullptr)
    {
        mlog()<<"getting world positions of world feature using get imo?"<<std::endl;
    }

    std::map<uint,Vector3d> xss;
    for(std::shared_ptr<Measurement>& m:ms){
        std::shared_ptr<TimePoint> tp=m->tp.lock();
        if(tp==nullptr) continue; // can measurements outlive their timepoints?
        PoseD p_c_world=*(tp->Pnw);
        auto Pimo=imo->getPose(tp->frameid);

        if(Pimo==nullptr){
            mlog()<<"feature has measurement older than the imo"<<std::endl;
            continue;
        }
        PoseD p_c_imo=*Pimo;
        Vector3d x_world=p_c_world.inverse()*p_c_imo*f->X;
        xss[tp->frameid]=x_world;
    }
    return xss;
}

std::vector<Vector3d> get_imo_feature_in_world(std::shared_ptr<Feature>& f, bool all=false){
    std::map<uint,Vector3d> xss=get_imo_feature_in_world_map(f);
    //cout<<"get_imo_feature_in_world - done"<<endl;
    std::vector<Vector3d> xs; xs.reserve(xss.size());
    for(auto x:xss)
        xs.push_back(x.second);
    // so thats the sorted list
    if(all)
        return xs;
    std::vector<Vector3d> tmp;
    for(uint i=xs.size()-2;i<xs.size();++i)
        tmp.push_back(xs[i]);
    return tmp;
}
template<class T> T mean(std::vector<T> ts){
    T t=ts.at(0);
    for(uint i=1;i<ts.size();++i)
        t+=ts[i];
    t/=double(ts.size());
    return t;
}
std::vector<Vector3d> get_imo_mean_in_world(std::vector<std::shared_ptr<Measurement>> ms){
    // all ms should have the same imo etc...
    std::vector<std::map<uint,Vector3d>> vs;vs.reserve(ms.size());
    for(std::shared_ptr<Measurement> m:ms)
        vs.push_back(get_imo_feature_in_world_map(m->feature));

    std::map<uint,std::vector<Vector3d>> map;
    for(std::map<uint,Vector3d> v:vs)
        for(auto t:v)
            map[t.first].push_back(t.second);
    // sorted by map
    std::vector<Vector3d> xs;
    for(std::pair<uint,std::vector<Vector3d>> e:map){
        xs.push_back(mean(e.second));
    }
    return xs;
}

std::vector<Flow> sequence_as_flow(std::vector<Vector3d> xs, Vector3d color){
    std::vector<Flow> flows;
    for(uint i=1;i<xs.size();++i){
        flows.push_back(Flow(xs[i-1], xs[i]- xs[i-1],color));
    }
    return flows;
}

std::vector<Flow> get_imo_curr_mean_flow(std::vector<std::shared_ptr<Measurement>> ms){
    // this thing shows how the mean of the current imo would have moved if it was tracked through its history.
    std::vector<Flow> flows;flows.reserve(ms.size());
    for(std::pair<std::shared_ptr<PoseImo>, std::vector<std::shared_ptr<Measurement>>> e:sort_by_imo(ms))
    {
        auto ms=e.second;



        // current imo center in world
        Vector3d xw=get_imo_mean_in_world(e.second).back();

        // compute the corresp xm
        std::shared_ptr<TimePoint> tp=ms.at(0)->tp.lock();

        PoseD Pnw=*(tp->Pnw);
        Vector3d xc=Pnw*xw;
        PoseD Pcm=*(e.first->getPose(e.first->get_frameid_zero()));// which must exist because m has it... not true!
        Vector3d xm=Pcm.inverse()*xc;


        auto ps=tp->map.lock()->get_poses();// Pcw

        // imo poses
        std::vector<PoseD> Pwms;
        std::map<int,PoseD> mPwms = e.first->get_poses();
        for(auto [time,pose]:mPwms){
            PoseD Pcm=pose;
            if(ps.find(time)!=ps.end()){
                PoseD Pcw=ps[time];
                Pwms.push_back(Pcw.inverse()*Pcm);
            }
        }

        std::vector<Vector3d> xs;
        for(uint i=std::max((int)0,((int)Pwms.size())-50);i<Pwms.size();++i)
            xs.push_back(Pwms[i]*xm);
        for(auto f:sequence_as_flow(xs,toV3(mlib::Color::nrlarge(2+(e.first->id % 32)))))
            flows.push_back(f);
    }
    return flows;
}

std::vector<Flow> get_imo_mean_flow(std::vector<std::shared_ptr<Measurement>> ms, [[maybe_unused]] bool all){

    std::vector<Flow> flows;flows.reserve(ms.size());
    for(std::pair<std::shared_ptr<PoseImo>, std::vector<std::shared_ptr<Measurement>>> e:sort_by_imo(ms)){
        auto tmp=sequence_as_flow(get_imo_mean_in_world(e.second),toV3(mlib::Color::nrlarge(2+(e.first->id % 32))));
        for(auto t:tmp)
            flows.push_back(t);
    }
    return flows;
}
PointCloud get_point_cloud(std::vector<std::shared_ptr<Measurement>> ms){

    PointCloud pc;
    pc.points.reserve(ms.size());
    for(std::shared_ptr<Measurement> m:ms)
    {
        if(m==nullptr)continue;
        if(m->getPrevious()==nullptr) continue;
        if(m->feature->get_imo()) continue;
        pc.points.push_back(m->feature->X);
    }
    pc.colors.resize(pc.points.size(),Vector3d(0,255,255));
    return pc;
}
std::shared_ptr<FlowField> get_imo_mean_flow_field(std::shared_ptr<TimePoint> tp){
    auto ff=std::make_shared<FlowField>();
    std::vector<std::shared_ptr<Measurement>> ms=not_nullptr_and_has_previous_filter(tp->ms);
    ff->flows=get_imo_curr_mean_flow(ms);
    ff->points=get_point_cloud(ms);
    ff->apply_transform((*(tp->Pnw)));
    return ff;
}

std::shared_ptr<FlowField> get_imo_flow_field(std::shared_ptr<TimePoint> tp){
    std::vector<std::shared_ptr<Measurement>> ms=not_nullptr_and_has_previous_filter(tp->ms);
    std::vector<Flow> flows;flows.reserve(ms.size());
    for(std::shared_ptr<Measurement>& m:ms){
        if(m->feature->get_imo()){
            // imos have their own coordinates system, this is related to the world one by the pose of any camera.
            // the latest may be the most accurate
            Vector3d color=toV3(mlib::Color::nrlarge(2+(m->feature->get_imo()->id % 32)));
            //color={0,0,0};
            auto xs=get_imo_feature_in_world(m->feature);
            std::vector<Flow> flow=sequence_as_flow(xs,color);
            for(Flow f:flow)
                flows.push_back(f);
        }
    }

    auto ff=std::make_shared<FlowField>();
    ff->flows=flows;
    ff->points=get_point_cloud(ms);
    ff->apply_transform((*(tp->Pnw)));
    return ff;
}
}

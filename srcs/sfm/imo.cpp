#include <cstdio>
#include <iostream>
#include <thread>
#include <experimental/filesystem>


#include <opencv2/highgui.hpp>

#include <mlib/utils/mlog/log.h>
#include <mlib/utils/cvl/triangulate.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/sfm/p3p/pnp_ransac.h>
#include <mlib/utils/cvl/triangulate_nl.h>
#include <sfm/map.h>
#include <sfm/imo.h>
#include <sfm/reprojection_error.h>
#include <sfm/measurement.h>
#include <sfm/calibration.h>

namespace fs=std::experimental::filesystem;
using std::cout;using std::endl;


namespace cvl{
std::vector<std::shared_ptr<Measurement>> not_nullptr_and_has_previous_and_not_imo_filter(std::vector<std::shared_ptr<Measurement>>& ms ){
    std::vector<std::shared_ptr<Measurement>> tmp;tmp.reserve(ms.size());
    for(auto& m:ms){

        if(m==nullptr) continue;
        if (m->getPrevious()==nullptr) continue;
        auto f=m->feature;
        if(!f->X.isnormal()) continue;
        if(f->get_imo()!=nullptr) continue;
        //if(f->size()<2) continue;

        tmp.push_back(m);
    }
    return tmp;
}

std::vector<std::shared_ptr<Measurement>> not_nullptr_filter(std::vector<std::shared_ptr<Measurement>>& ms ){
    std::vector<std::shared_ptr<Measurement>> tmp;tmp.reserve(ms.size());
    for(auto& m:ms)
        if(m)
            tmp.push_back(m);
    return tmp;
}
std::vector<std::shared_ptr<Measurement>> not_nullptr_and_has_previous_filter(std::vector<std::shared_ptr<Measurement>>& ms ){
    std::vector<std::shared_ptr<Measurement>> tmp;tmp.reserve(ms.size());
    for(auto& m:ms)
        if(m)
            if(m->getPrevious()){
                if(m->feature==nullptr){cout<<"WTF no featuere?"<<endl; exit(1);}
                tmp.push_back(m);
            }
    return tmp;
}
std::vector<std::shared_ptr<Measurement>> not_nullptr_and_has_previous_and_is_inlier_filter(std::vector<std::shared_ptr<Measurement>>& ms ,double max_err){
    std::vector<std::shared_ptr<Measurement>> tmp;tmp.reserve(ms.size());
    for(auto& m:ms)
        if(m)
            if(m->getPrevious()){
                if(m->feature->error()<max_err)
                    tmp.push_back(m);
            }
    return tmp;
}
std::vector<std::shared_ptr<Measurement>> not_nullptr_and_has_previous_and_imo_filter(std::vector<std::shared_ptr<Measurement>>& ms ){
    std::vector<std::shared_ptr<Measurement>> tmp;tmp.reserve(ms.size());
    for(auto& m:ms)
        if(m)
            if(m->getPrevious()){
                if(m->feature==nullptr){cout<<"WTF no featuere?"<<endl; exit(1);}
                if(m->feature->get_imo()!=nullptr)
                    tmp.push_back(m);
            }
    return tmp;
}
std::vector<std::shared_ptr<Measurement>> not_nullptr_and_has_previous_and_is_car(std::vector<std::shared_ptr<Measurement>>& ms ){
    std::vector<std::shared_ptr<Measurement>> tmp;tmp.reserve(ms.size());
    for(auto& m:ms)
        if(m)
            if(m->getPrevious()){
                if(m->feature==nullptr){cout<<"WTF no featuere?"<<endl; exit(1);}
                if(m->label>0)
                    tmp.push_back(m);
            }
    return tmp;
}
std::map<std::shared_ptr<PoseImo>,
std::vector<std::shared_ptr<Measurement>>> sort_by_imo(std::vector<std::shared_ptr<Measurement>>& ms){
    std::map<std::shared_ptr<PoseImo>, std::vector<std::shared_ptr<Measurement>>> out;
    for(std::shared_ptr<Measurement>& m:ms){
        if(m==nullptr) continue;
        auto imo=m->feature->get_imo();
        if(imo){ // not null
            auto& tmp=out[imo];
            tmp.reserve(ms.size());
            tmp.push_back(m);
        }
    }
    return out;
}

void printerrs(std::vector<Vector3d> xs, std::vector<Vector2d> yns, PoseD Pin){

    std::vector<float> errs;
    for(uint i=0;i<xs.size();++i){
        float err=(HirCalibration::common().project(Pin*xs[i]) -
                   HirCalibration::common().distort(yns[i])).norm();
        err=std::min(err,99.0f);
        errs.push_back(err);
        //err+=(K*(Prw*xs[i]).dehom() - K*yrns[i]).length();
    }
    std::sort(errs.begin(),errs.end());
    int i=0;
    for(float err:errs){
        std::printf("%2.1f   ",err);
        if((i++ % 15)==9) cout<<"\n";
    }
    cout<<"\ntotal: "<<errs.size()<<endl;
}
uint get_inliers(std::vector<Vector3d> xs, std::vector<Vector2d> yns, PoseD Pin, double threshold){
    uint inliers=0;
    for(uint i=0;i<xs.size();++i){

        double err=(HirCalibration::common().project(Pin*xs[i]) -
                    HirCalibration::common().distort(yns[i])).norm();
        if(err<threshold)
            inliers++;
    }
    return inliers;
}

int IMO::age(){
    return ps.size();
}
PoseD IMO::delta_pose(){
    PoseD pose0,pose1;
    // relies on map giving me them in order? yes
    for([[maybe_unused]] auto [key,p]:ps){
        pose0=pose1;
        pose1=*p;
    }
    return pose1*pose0.inverse();
}

int IMO::count=1;







std::vector<std::shared_ptr<IMO> > getImos(std::vector<std::shared_ptr<Measurement>>& ms){
    std::set<std::shared_ptr<IMO> > unique;
    for(auto& m:ms) unique.insert(m->feature->get_imo());
    std::vector<std::shared_ptr<IMO>> ret;
    for(auto u:unique) ret.push_back(u);
    return ret;
}








std::shared_ptr<PoseImo> failure_to_create(std::vector<std::shared_ptr<Measurement>>& ms){
    for(auto& m:ms){
        m->feature->clear_imo();
        //m->feature->split_track();
    }
    return nullptr;
}

std::shared_ptr<PoseImo>
PoseImo::create(std::vector<std::shared_ptr<Measurement>>& ms, uint min_creation_count){

    // so basically the ms are outright outliers to everything else at this point!
    // thus, toss everything except the two last observations for each such feature
    // then triangulate in the previous one,
    // then pnp to the new one
    bool display_errors=true;

    for(std::shared_ptr<Measurement>& m:ms)
    {
        if(m->feature->get_imo()!=nullptr)
            cout<<"hmm??? already imo?"<<endl;
        //m->feature->split_track(); // should be size 2 so this is free
        m->feature->X=m->getPrevious()->triangulate(); // so previous frame is identity!
        // not really needed !
        assert(m->feature->size()>1);
    }

    if(false)
    {
        BoundingBox bb=get_bounding_box(ms);
        if (bb.area()< 40*40 || bb.rows()<30 || bb.cols()<30) return failure_to_create(ms);
    }


    cout<<"Creating IMO using  "<<ms.size() <<" candidates "<<endl;






    // all in the same time
    uint time0=ms.at(0)->getPrevious()->frameid;
    uint time1=ms.at(0)->frameid;


    std::vector<Vector3d> xs;   xs.reserve(ms.size());
    std::vector<Vector2d> yns; yns.reserve(ms.size());
    for(auto& m:ms){
        Vector3d Xworld=m->feature->X;
        xs.push_back(Xworld);
        yns.push_back(m->ynl());
    }

    if(display_errors)
    {
        cout<<"before: "<<endl;
        printerrs(xs,yns,PoseD());
        cout<<"\n";
    }

    cvl::PnpParams params;params.threshold=1.5/HirCalibration::common().fx();
    cvl::PoseD Pin=cvl::pnp_ransac(xs, yns, params);

    if(display_errors){
        // test
        cout<<"after: "<<endl;
        printerrs(xs,yns,Pin);
        cout<<"\n"<<endl;
    }
    // check inliers
    uint inliers=get_inliers(xs,yns,Pin,2);
    if(inliers<min_creation_count){
        cout<<"failed due to too few inliers"<<inliers<<endl;
        return failure_to_create(ms);
    }
    std::shared_ptr<PoseImo> imo=std::make_shared<PoseImo>();
    imo->self=imo;
    auto origin=std::make_shared<PoseD>(PoseD());
    imo->set_pose(time0, origin);//imo->ps[time0]=origin; // this is the map of poses where the imo has been found...
    imo->origin=origin;
    imo->set_pose(time1, std::make_shared<PoseD>(Pin));

    // this refines all the features...
    for(std::shared_ptr<Measurement>& m:ms){
        imo->getMinReprojectionDistance(m->feature);
    }
    if(display_errors){
        xs.clear();
        for(uint i=0;i<ms.size();++i)
            xs.push_back(ms[i]->feature->X);
        cout<<"after refineing2 "<<endl;
        printerrs(xs,yns,Pin);
    }



    std::vector<std::shared_ptr<Measurement>> minliers;minliers.reserve(ms.size());

    for(auto& m:ms){

        m->feature->set_imo(imo);

        double err=m->feature->error();
        if(err<2){
            minliers.push_back(m);
        }else
            m->feature->clear_imo();
    }
    cout<<"done with reprojection check: "<<minliers.size()<<endl;
    if(minliers.size()<min_creation_count)
        return failure_to_create(ms);
    {


        // problem! the threshold here needs to be adaptive to the distance to the object!

        Vector3d xm=median_centroidvs(minliers);

        std::vector<std::shared_ptr<Measurement>> minliers2;
        minliers2.reserve(minliers.size());

        for(std::shared_ptr<Measurement>& m:minliers){
            //double thr=std::max(10.0, xm.norm()*0.25);
            if((m->feature->X - xm).length()>10){
                m->feature->clear_imo();
                // maybe clear hard here
                //m->feature->split_track(1);

            }else
                minliers2.push_back(m);
        }
        minliers=minliers2;
        cout<<"done with center mass check!: "<<minliers.size()<<endl;


    }

    if(minliers.size()<min_creation_count){
        cout<<"too few inliers in hir: "<<minliers.size()<<endl;
        return failure_to_create(ms);
    }
    {
        BoundingBox bb=get_bounding_box(minliers);
        if (bb.area()< 50*50 || bb.rows()<40 || bb.cols()<40) return failure_to_create(ms);
    }
    if(true){
        PoseD P0=*ms.at(0)->getPrevious()->tp.lock()->Pnw;
        PoseD P1=*ms.at(0)->tp.lock()->Pnw;
        PoseD P10=P1*P0.inverse();
        cout<<"moved distance: "<<(Pin*P10.inverse()).translation().length()*30*3.6<<"km/h"<<endl;
        if((Pin*P10.inverse()).translation().length()*30*3.6<5) // corresponds to a speed of what?
            return failure_to_create(ms);
    }
    imo->id=PoseImo::count++;


    cout<<"Create: "<<imo->id<<" done with "<<minliers.size()<<endl;



    // find earlier poses?



    return imo;
}
Vector3d PoseImo::find_initial_X_for(std::shared_ptr<Feature>& f) const{
    std::vector<std::weak_ptr<Measurement>> ms=f->ms;
    for(auto& wm:f->ms)
    {
        auto m=wm.lock();
        if(!m) continue;
        auto it=getPose(m->frameid);

        if(it!=nullptr){
            return (*it).inverse()*m->triangulate();
        }
    }
    cout<<"something very wierd234fkljagljs"<<endl;;
    return Vector3d(0,0,0);
}

double PoseImo::getMinReprojectionDistance(std::shared_ptr<Feature>& f) const
{
    f->clear_imo();
    //cout<<"getMinReprojectionDistance: "<<endl;
    //std::vector<std::shared_ptr<Measurement>> ms=f->getMeasurements();
    std::vector<std::shared_ptr<Measurement>> ms=f->get_last_n(3);
    // better init always helps...
    if(ms.size()<2) {
        mlog()<<"checked min reprojection distance for imo for feature of size 1!"<<endl;

        return 0;
    }


    bool use_ceres=true; // slower but might be better...
    if(use_ceres){
        auto sself=self.lock();

        f->X=find_initial_X_for(f);
        ceres::Problem problem;

        for(std::shared_ptr<Measurement>& m:ms)
        {
            std::shared_ptr<PoseD> pose= getPose(m->frameid);
            if(pose==nullptr) continue;
            problem.AddResidualBlock(StereoTriangulationError::Create(
                                         pose->get4x4(),m->obs()),
                                     nullptr,
                                     &f->X[0]);
        }
        ceres::Solver::Options options;{
            options.linear_solver_type = ceres::DENSE_QR;// default, good
            options.max_num_iterations=6;
        }
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //cout<<"Full Summary: \n"<<summary.FullReport()<<endl;


        return f->error(self.lock());
    }else{

        std::vector<PoseD> as;as.reserve(ms.size()*2);
        std::vector<Vector2d> ys;ys.reserve(ms.size()*2);
        for(std::shared_ptr<Measurement>& m:ms){
            std::shared_ptr<PoseD> pose= getPose(m->frameid);
            if(pose==nullptr) continue;
            ys.push_back(m->ynl());
            ys.push_back(m->ynr());
            PoseD Plw=*pose;
            PoseD Prw=Plw;
            Prw.t[0]-=HirCalibration::common().baseline();
            as.push_back(Plw);
            as.push_back(Prw);
        }
        if(ys.size()<2){
            mlog()<<"warning, this shouldnt happen"<<endl;
            return 0;
        }
        f->X=triangulate_nl<double>(ys,as);
        return f->error();
    }
}

bool test_or_fix(std::shared_ptr<Feature>& f, double thr){
    if(f->error()<thr) return true;
    auto f2=f->split_track();
    f2->refine(true);
    if(f->error()<thr)return true;
    f2->clear_imo();
    return false;
}

bool failure2estimate(std::vector<std::shared_ptr<Measurement>>& ms){
    for(auto& m:ms){
        m->feature->ms.pop_back();
        m->feature=Feature::create(Vector3d(0,0,0));
        m->feature->add(m);
        m->feature->refine(true);
    }
    return false;
}

bool PoseImo::estimate(std::vector<std::shared_ptr<Measurement>>& ms,
                       int min_inlier_count,
                       double reprojection_threshold){
    mlib::Timer timer("pose imo estimate");
    timer.tic();
    cout<<"IMO Estimate: tracks "<<ms.size()<<" imo nr: "<<id<<endl;

    //standard pnp
    //std::map<int, PoseD> ps=map->get_poses();
    if(int(ms.size())<min_inlier_count) {
        //died due to beeing too small...
        cout<<"IMO Lost"<<endl;
        return failure2estimate(ms);
    }
    cout<<"IMO PREFILTER... "<<endl;

    uint time=ms[0]->tp.lock()->frameid;
    std::vector<Vector3d> xs;xs.reserve(ms.size());
    std::vector<Vector2d> yns; yns.reserve(ms.size());
    // all in the same time
    for(uint i=0;i<ms.size();++i)
    {
        Vector3d Xworld=ms[i]->feature->X;
        Vector2d yn=ms[i]->ynl();
        xs.push_back(Xworld);
        yns.push_back(yn);
    }
    cvl::PnpParams params;
    params.max_iterations=1000;
    params.min_iterations=200;
    params.threshold=reprojection_threshold/HirCalibration::common().fx();
    cvl::PoseD Pin=cvl::pnp_ransac(xs, yns,params);



    // test
    cout<<"imo pnp errors: "<<endl;
    printerrs(xs,yns,Pin);
    cout<<"\n"<<endl;
    set_pose(ms.at(0)->frameid, std::make_shared<PoseD>(Pin));

    // lets have a look at the number of inliers
    // basically this may change older tracks, which is kindof pointless probably..
    int inliers=0;
    for(auto& m:ms)
        if(test_or_fix(m->feature,reprojection_threshold))
            inliers++;
    // remove a point that is too far from the median
    Vector3d xm=median_centroidvs(xs);
    for(auto& m:ms)
        if((m->feature->X - xm).norm()>std::max(10.0,xm.norm()/2.0))
            m->feature->clear_imo();



    if(inliers<min_inlier_count){
        remove_pose(time);
        return failure2estimate(ms);
    }
    // this might reduce the nr of outliers, but not increase it I think...
    refine(reprojection_threshold);


    bbs[time]=get_imo_bounding_box(time,true);


    return true;
}



void PoseImo::refine(double reprojection_threshold)
{
    if(age()<2) return;
    mlog()<<"refine"<<endl;




    problemtimer.tic();
    //get last N timepoints, step three get all ms with this imo from that tp
    std::vector<std::shared_ptr<TimePoint>> tps=map->get_last_n_timepoints(4);
    {
        std::vector<std::shared_ptr<TimePoint>> tmp;tmp.reserve(tps.size());
        for(auto& tp:tps)
            if(has_pose(tp->frameid))
                tmp.push_back(tp);
        tps=tmp;
    }

    if(tps.size()<2) return;
    // get meas with this imo for each

    // not incremental first!
    ceres::Problem problem;
    // start adding poses at
    auto sself=self.lock();
    std::shared_ptr<PoseD> first=nullptr;
    std::set<int> founds;
    for(auto& tp:tps)
    {
        int frameid=tp->frameid;
        std::shared_ptr<PoseD> pose= getPose(frameid);
        for(auto& m:tp->ms){
            auto& f=m->feature;
            if(f->get_imo()!=sself) continue;
            if(f->error()>reprojection_threshold){
                continue;
            }
            if(first==nullptr){first=pose;}
            auto resid=StereoReprojectionError::Create(m->obs());
            problem.AddResidualBlock(resid,nullptr,
                                     pose->getRRef(),
                                     pose->getTRef(),
                                     f->X.begin());
            founds.insert(frameid);
        }
    }
    if(founds.empty()) {
        mlog()<<"strange?\n";
        return;
    }
    problem.SetParameterBlockConstant(first->getRRef());
    problem.SetParameterBlockConstant(first->getTRef());

    for(int found:founds)
    {
        auto pose=getPose(found);
        ceres::LocalParameterization* qp = new ceres::QuaternionParameterization;
        problem.SetParameterization(pose->getRRef(), qp);
    }
    problemtimer.toc();
    batimer.tic();

    ceres::Solver::Options options;{
        options.linear_solver_type = ceres::SPARSE_SCHUR;// default, good
        options.max_num_iterations=5;
        options.num_threads=std::thread::hardware_concurrency();
    }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout<<"Full Summary for imo: "+str(id)+ "\n"<<summary.FullReport()<<endl;
    batimer.toc();

    cout<<"imo: "<<id<<" "<<problemtimer<< " "<<batimer<<endl;
    mlog()<<"refine done"<<endl;
}


PoseImo::PoseImo(){}
PoseImo::~PoseImo(){}



Vector3d median_centroidvs(std::vector<Vector3d>& vs){
    if(vs.size()<1)
        return Vector3d(0,0,0);
    std::vector<double> xs,ys,zs;
    xs.reserve(vs.size());
    ys.reserve(vs.size());
    zs.reserve(vs.size());
    for(auto& v:vs){
        xs.push_back(v[0]);
        ys.push_back(v[1]);
        zs.push_back(v[2]);
    }

    // median for x, y, z.
    sort(xs.begin(), xs.end());
    sort(ys.begin(), ys.end());
    sort(zs.begin(), zs.end());
    Vector3d xm(xs.at(xs.size()/2),
                ys.at(ys.size()/2),
                zs.at(ys.size()/2));
    return xm;
}
Vector3d median_centroidvs(std::vector<std::shared_ptr<Measurement>>& vs){
    std::vector<Vector3d> xs;xs.reserve(vs.size());
    for(auto& m:vs)
        xs.push_back(m->feature->X);
    return median_centroidvs(xs);
}

Vector3d PoseImo::median_centroid(){
    std::vector<Vector3d> vs;
    for(auto& feature:features){
        Vector3d x=feature->X;
        if(x.isnormal()){
            vs.push_back(x);
        }
    }
    return median_centroidvs(vs);
}
Vector3d PoseImo::get_mass_center(){
    Vector3d xm(0,0,0);
    for(auto& feature:features){
        auto x=feature->X;
        if(x.isnormal() && x.length()<1e3)
            xm+=feature->X;
    }
    if(features.size()>0)
        xm/=features.size();
    return xm;
}
void PoseImo::add_feature(std::shared_ptr<Feature>& f){
    if(map==nullptr)
        map=f->ms.at(0).lock()->tp.lock()->map.lock();
    features.insert(f);
}
void PoseImo::remove_feature(std::shared_ptr<Feature>& f){
    features.erase(f);
}



std::vector<std::shared_ptr<Measurement>> PoseImo::get_measurements_in_frame(int frame_id) const{
    std::vector<std::shared_ptr<Measurement>> ms;
    auto tp=map->at_frameid(frame_id);
    if(tp==nullptr) return ms;
    ms.reserve(tp->ms.size());
    auto sself=self.lock();
    for(auto& m:tp->ms){
        if(m->feature->get_imo()==sself)
            ms.push_back(m);
    }
    return ms;
}

BoundingBox PoseImo::get_imo_bounding_box(int frame_id, bool conservative) const{
    auto it=bbs.find(frame_id);
    if(it!=bbs.end()) return it->second;
    auto ms=get_measurements_in_frame(frame_id);
    // filter measurements for so its only the longer ones?
    auto bb=get_bounding_box(ms, conservative); bb.id=id;
    return bb;
}

imores PoseImo::get_imo_res(){

    imores ir;
    ir.imo_id=id;
    ir.xm = median_centroid();
    ir.res.reserve(age());
    auto poses=get_poses();
    cout<<"imo_res: "<<id<<" "<<poses.size()<<endl;
    for(auto [frame_id, pose]:poses){

        auto ms=get_measurements_in_frame(frame_id);
        // filter measurements for so its only the longer ones?
        auto bb=get_imo_bounding_box(frame_id,true);
        auto fxm = median_centroidvs(ms);
        ir.res.push_back(imoframeres(frame_id, pose, fxm, bb));
    }
    return ir;
}

}

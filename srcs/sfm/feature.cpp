#include <iostream>
#include <ceres/ceres.h>

#include <mlib/utils/mlog/log.h>
#include <mlib/utils/cvl/triangulate.h>
#include <mlib/utils/cvl/triangulate_nl.h>

#include <sfm/feature.h>
#include <sfm/reprojection_error.h>
#include <sfm/imo.h>
#include <sfm/measurement.h>


using std::endl;using std::cout;

namespace cvl{
std::shared_ptr<Feature> Feature::create(const Vector3d& x){
    auto ptr=std::make_shared<Feature>();
    ptr->self=ptr;
    ptr->X=x;
    ptr->ms.reserve(50);
    return ptr;
}
void Feature::add(std::shared_ptr<Measurement>& m){
    assert(m!=nullptr);
    ms.push_back(m);
}

template<class T> std::vector<T> reverse(std::vector<T>& v){
    std::vector<T> rs; rs.reserve(v.size());
    for(auto it=v.rbegin();it!=v.rend();++it)
        rs.push_back(*it);
    return rs;
}
std::vector<std::shared_ptr<Measurement>>
Feature::get_last_n(uint n)  const
{
    std::vector<std::shared_ptr<Measurement>> out;out.reserve(ms.size());
    for(auto it=ms.rbegin();it!=ms.rend() && out.size()<=n;++it){
        auto m=it->lock();
        if(m)out.push_back(m);
    }
    return reverse(out);
}
std::vector<std::shared_ptr<Measurement> > Feature::getMeasurements(){
    std::vector<std::shared_ptr<Measurement>> tmp;tmp.reserve(ms.size());
    for(auto& m:ms){
        auto mp=m.lock();
        if(mp)
            tmp.push_back(mp);
    }
    ms.clear();
    for(auto& m:tmp)
        ms.push_back(m);
    return tmp;
}
// from previous into current!
Vector3d  Feature::world_velocity(){
    if(imo==nullptr) return Vector3d(0,0,0);
    // cout<<"get_imo_feature_in_world"<<endl;
    std::vector<std::shared_ptr<Measurement>> sms = get_last_n(2);

    if(sms.size()<2) return Vector3d(0,0,0);



    // for each measurement get the imo and timepoint poses


    auto tp0=sms[0]->tp.lock();
    auto tp1=sms[1]->tp.lock();
    if(tp0==nullptr||tp1==nullptr){
        mlog()<<"strange\n";
        return Vector3d(0,0,0);
    }

    PoseD prev_p_c_world=*(tp0->Pnw);
    PoseD curr_p_c_world=*(tp1->Pnw);

    if(imo->getPose(sms[0]->frameid)==nullptr || imo->getPose(sms[1]->frameid)==nullptr){
        mlog()<<"missing time in imo"<<std::endl;
        return Vector3d(0,0,0);
    }

    PoseD prev_p_c_imo= *imo->getPose(sms[0]->frameid);

    if(imo->getPose(sms[1]->frameid)==nullptr){
        mlog()<<"warning, something weird"<<std::endl;
    }

    PoseD curr_p_c_imo= *imo->getPose(sms[1]->frameid);


    Vector3d prev_x=prev_p_c_world.inverse()*prev_p_c_imo*X;
    Vector3d curr_x=curr_p_c_world.inverse()*curr_p_c_imo*X;
    cout<<"gotten world velocity in feature"<<endl;
    return curr_x - prev_x;
}


void Feature::clear()
{
    std::vector<std::weak_ptr<Measurement>> tmp;tmp.reserve(ms.size());
    for(auto& m:ms){
        auto mp=m.lock();
        if(mp)
            tmp.push_back(mp);
    }
    ms=tmp;
}

std::shared_ptr<Feature> Feature::split_track(){

    // TODO, verify results, after the added splits!

    // we do this for tracks that are outliers!
    // lets assert contiguous observations...
    // then, for tracks longer than three,
    //


    auto sms=get_last_n(3); // important for speed!
    if(sms.size()<3) return self.lock();



    // now pop the last two and create new ones!

    std::shared_ptr<Measurement> m=sms[sms.size()-2];
    std::shared_ptr<Measurement> n=sms[sms.size()-1];
    // old feature will lose n
    ms.pop_back();// drops one!



    std::shared_ptr<Measurement> copy=m->tp.lock()->createMeasurement(m->yl(),m->disparity,m->label);
    m->tp.lock()->ms.push_back(copy); // would be cleared away by anms, but we dont track this frame again...

    n->feature=copy->feature;
    n->feature->add(n);
    n->feature->refine(true);
    return n->feature;
}
void Feature::split_long_track(){

    //cout<<"splitting track!"<<endl;

    if(ms.size()<8) return;
    auto sms=getMeasurements();
    if(sms.size()<8) return;


    // now split into two,
    // first is [0,5], second is [5],10
    // 9 is still alot...


    ms.resize(4);
    /* cant do this while tracker is integrated in timepoint ...
    auto tp=sms[5]->tp.lock();
    if(tp){// technically guaranteed, but lets be careful...
        auto m=tp->createMeasurement(sms[5]->ynl,sms[5]->ynr,sms[5]->label,sms[5]);
        ms.push_back(m);
        tp->add_extra_measurement(m);
    }
    */

    std::vector<std::shared_ptr<Measurement>> bs;

    bs.reserve(16);
    for(uint i=4;i<sms.size();++i)
        bs.push_back(sms[i]);

    sFeature f=Feature::create(X);
    for(auto& b:bs){
        b->feature=f;
        f->add(b);
    }
    if(imo!=nullptr)
        f->set_imo(imo);
}
uint Feature::size(){    return ms.size();  }
uint Feature::max_num=3;
std::vector<double> Feature::errors(const std::shared_ptr<PoseImo>& imo) const{
    auto sms=get_last_n(max_num);
    std::vector<double> errs; errs.reserve(sms.size());
    for(auto& m:sms)
    {
        auto sp=imo->getPose(m->frameid);
        if(sp)
        {
            double err=(m->obs() - HirCalibration::common().stereo_project((*sp)*X)).norm();
            if(!HirCalibration::common().behind_either((*sp)*X))
                errs.push_back(err);
            else
                errs.push_back(100);
        }
    }
    return errs;
}
std::vector<double> Feature::errors() const
{

    auto sms=get_last_n(max_num);
    std::vector<double> errs; errs.reserve(sms.size());
    if(imo) return errors(imo);

    for(auto& m:sms)
    {
        PoseD p=m->get_pose();
        double err=(m->obs() - HirCalibration::common().stereo_project((p)*X)).norm();
        if(!HirCalibration::common().behind_either((p)*X))
            errs.push_back(err);
        else
            errs.push_back(100);
    }

    return errs;
}
double error_selected(const std::vector<double>& errs){
    if(errs.size()==0){
        cout<<"checking error on empty feature?"<<endl;
        return 0;
    }
    double m=0;
    for(auto e:errs)
        m+=e;

    return m/double(errs.size());
    /*
     *     double mv=0;
    for(double e:errs)
        mv=std::max(mv,e);
    return mv;
    */
}
Vector3d stereo_project();
double Feature::error() const
{
    return error_selected(errors());

}
double Feature::error(const std::shared_ptr<PoseImo>& imo) const
{
    return error_selected(errors(imo));
}

void Feature::refine(bool reinitialize) {
    if(ms.size()==1)
        if(reinitialize) X=ms[0].lock()->triangulateWorld();

    // getting the poses is expensive, so lets only do that beforehand...

    if(imo!=nullptr) { // generally not needed separately as they are always bundled.
        auto sself=self.lock();
        assert(sself);
        imo->getMinReprojectionDistance(sself);
        return;
    }
    // cout<<"Pre refine X"<<X<<endl;

    std::vector<std::shared_ptr<Measurement>> sms=get_last_n(3);
    if(sms.size()<2) return;
    //Vector3d prev_x=X;
    if(reinitialize) X=sms[0]->triangulateWorld();

    bool use_ceres=false;
    if(use_ceres){

        ceres::Problem problem;

        for(auto& m:sms)
            //for(int i=std::max(0,int(sms.size())-5);i<int(sms.size());++i)
        {
            //auto& m=sms[i];
            problem.AddResidualBlock(StereoTriangulationError::Create(m->tp.lock()->Pnw->get4x4(),
                                                                      m->obs()), nullptr, X.begin());
        }
        //cout<<"built problem?"<<endl;
        ceres::Solver::Options options;{
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations=10;
        }
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        /*
        double perr=0;
        double err=0;
        double count=0;
        for(std::shared_ptr<Measurement>& m:sms)
        {
            PoseD P = m->tp.lock()->pose();
            perr+=(m->obs() - HirCalibration::common().stereo_project(P*prev_x)).norm();
            err +=(m->obs() - HirCalibration::common().stereo_project(P*X)).norm();
count++;
        }
        cout<<"pre  reproj: "<<perr/count<<endl;
        cout<<"post reproj: "<<err/count<<endl;
        cout<<"mean"<< meanReprojectionError()<<endl;
*/
    }

    else{

        //cout<<"REIMPLEMENT!"<<endl;

        std::vector<PoseD> ps;ps.reserve(2*sms.size());
        std::vector<Vector2d> ys;ys.reserve(2*sms.size());
        for(auto& m:sms){
            ys.push_back(m->ynl());
            ys.push_back(m->ynr());
            std::shared_ptr<TimePoint> tp=m->tp.lock();
            ps.push_back(*tp->Pnw);
            ps.push_back(HirCalibration::common().Prl()*(*tp->Pnw));
        }
        //refine_timer.tic();
        if(reinitialize)
            X=triangulate_nl<double>(ys,ps);// dont reset the feature unless its bad!
        else
            X = gn_minimize(ys,ps,X);
        //refine_timer.toc();

        if(!X.isnormal())
            X=triangulate(PoseD(ps[0]),PoseD(ps[1]),Vector2d(ys[0]),Vector2d(ys[1]));
        if(!X.isnormal())
            X=Vector3d(0,0,1);

    }

    //cout<<refine_timer<<endl;
}

std::shared_ptr<PoseImo> Feature::get_imo(){
    return imo;
}
bool Feature::has_imo() const{
    return imo!=nullptr;
}

void Feature::set_imo(std::shared_ptr<PoseImo>& new_imo){
    if(imo==nullptr && new_imo==nullptr){
        cout<<"setting to the world, which it allready was"<<endl;
        return;
    }
    if(imo==new_imo){
        //cout<<"setting to the imo it allready was"<<endl;
        return;
    }
    auto f= self.lock();
    if(imo!=nullptr){
        cout<<"swapping from one imo, to the next without intermediate?"<<endl;
        imo->remove_feature(f);
    }
    imo=new_imo;
    if(imo!=nullptr)
        imo->add_feature(f);
}
std::shared_ptr<Feature> Feature::get_self(){return self.lock();}
int Feature::last_ms_frameid() const{
    if(ms.size()==0) return -1;
    auto m = ms.back().lock();
    if(m)
        return m->frameid;
    return -1;
}
int Feature::imo_age(){

    int age=0;
    for(auto& m:getMeasurements())
        if(imo->has_pose(m->frameid))
            age++;
    return age;
}
void Feature::clear_imo(){
    if(!imo) return;
    auto f=self.lock();
    assert(f);
    if(imo)
        imo->remove_feature(f);
    imo=nullptr;
    f->refine(true);
}
std::shared_ptr<Measurement> Feature::get_measurement_in_frame(uint frame_id){
    for(auto& m:getMeasurements())
        if(m->frameid==frame_id) return m;
    return nullptr;
}

}// end namespace cvl

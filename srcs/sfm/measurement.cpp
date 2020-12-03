#include <iostream>
#include <mlib/utils/cvl/triangulate.h>

#include <sfm/measurement.h>
#include <sfm/feature.h>
#include <sfm/imo.h>
#include <sfm/calibration.h>

using std::cout;
using std::endl;
namespace cvl {




Vector2d Measurement::yl() const{ return y;}
Vector2d Measurement::yr() const{return y+Vector2d(0,-disparity);}
Vector2d Measurement::ynl() const{ return HirCalibration::common().undistort(y);}
Vector2d Measurement::ynr() const{return HirCalibration::common().undistort(y+Vector2d(0,-disparity));}
PoseD Measurement::get_pose()
{
    std::shared_ptr<TimePoint> t=tp.lock();
    assert(t!=nullptr);
    return *t->Pnw;
}
Vector3d Measurement::world_velocity()
{

    //if(feature->get_imo()==nullptr)

    return Vector3d(0,0,0);
    // TODO implement!
}
double Measurement::reprojectionErrorWorld(){
    return reprojectionError(*(tp.lock()->Pnw));
}

double Measurement::reprojectionError(PoseD P,
                                      Vector3d x){
    cvl::Vector3d X=P*x;

    if(HirCalibration::common().behind_either(X)){
        //cout<<"behind"<<endl;
        return 1000;
    }
    Vector3d obsr= HirCalibration::common().stereo_project(X);
    if(obsr[2]<0){
        cout<<"now this is wierd!"<<endl;
        exit(1);
    }
    //cout<<"obs: " << obsr <<" " << this->obs()<<endl;
    auto diff=obsr - obs();
    diff[2]*=0.5;
    return diff.norm();
}
double Measurement::error(){
    auto imo=feature->get_imo();
    PoseD P;
    if(imo!=nullptr){
        auto p=imo->getPose(frameid);
        if(p==nullptr) return 100;
        P=*p;
    }
    else
        P=*(tp.lock()->Pnw);
    return reprojectionError(P,feature->X);
}
double Measurement::reprojectionError(cvl::PoseD Plw){
    return reprojectionError(Plw, feature->X);
}

cvl::Vector3d Measurement::triangulate() const
{
    return HirCalibration::common().triangulate_ray(y,disparity).dehom();
}
cvl::Vector3d Measurement::triangulateWorld() const{
    return tp.lock()->Pnw->inverse()*triangulate(); // very slow!
}
double Measurement::motionDisparity(){

    if(!getPrevious()) return 0;
    return (y-getPrevious()->y).norm();
}


std::shared_ptr<Measurement> Measurement::getPrevious(){
    if(feature->size()<2) return nullptr;
    return feature->ms.at(feature->ms.size()-2).lock();
}





BoundingBox get_bounding_box(std::vector<std::shared_ptr<Measurement>>& ms, bool conservative){
    //asserts that they are not null!
    if(ms.size()==0) return BoundingBox(-1,0,0,0,0);
    double row_start=ms[0]->yl()[0];
    double row_end=ms[0]->yl()[0];
    double col_start=ms[0]->yl()[1];
    double col_end=ms[0]->yl()[1];

    if(!conservative){

        // if more than half are older than 2 on the imo use only the twos
        for(auto& m:ms){
            if(m->feature->size()<2) continue;
            if(m->error()>3) continue;



            auto y=m->yl();
            double r=y[0];
            double c=y[1];
            row_start=std::min(row_start, r);
            col_start=std::min(col_start, c);
            row_end=std::max(row_end, r);
            col_end=std::max(col_end, c);
        }
        return BoundingBox(-1,row_start, col_start, row_end, col_end);
    }



    // if more than half are older than 2 on the imo use only the twos

    int count=0;
    for(auto& m:ms){
        if(m->feature->size()<2) continue;
        if(m->error()>3) continue;
        if(m->feature->imo_age()>2)           count++;    }

    std::vector<double> rs,cs;
    rs.reserve(ms.size());cs.reserve(ms.size());
    for(auto& m:ms)
    {
        if(m->feature->size()<2) continue;
        if(m->error()>3) continue;

        if( count>ms.size()/2.0){
            //cout<<"skipping in boundingbox "<<count<<endl;
            if(m->feature->imo_age()<3){
                continue;
            }
        }



        auto y=m->yl();
        double r=y[0];
        double c=y[1];
        row_start=std::min(row_start, r);
        col_start=std::min(col_start, c);
        row_end=std::max(row_end, r);
        col_end=std::max(col_end, c);

        rs.push_back(r);
        cs.push_back(c);

    }
    std::sort(rs.begin(),rs.end());
    std::sort(cs.begin(),cs.begin());

    // remove the 5 or 5% most extreme

    if(rs.size()>20 && false){
        int remove = 2.0;
        std::vector<double> trs; trs.reserve(rs.size());
        std::vector<double> tcs; tcs.reserve(rs.size());
        for(int i=remove;i<int(rs.size())-remove;++i){
            trs.push_back(rs[i]);
            tcs.push_back(cs[i]);
        }

        row_start=trs[0];
        row_end=trs.back();
        col_start=tcs[0];
        col_end=tcs.back();
    }


    return BoundingBox(-1,row_start, col_start, row_end, col_end);
}

}

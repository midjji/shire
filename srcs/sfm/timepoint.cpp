#include <iostream>

#include <mlib/utils/cvl/triangulate.h>

#include <sfm/feature.h>
#include <sfm/map.h>
#include <sfm/timepoint.h>
#include <sfm/measurement.h>


using std::cout;
using std::endl;
namespace cvl {


std::shared_ptr<Measurement> TimePoint::createMeasurement(Vector2d ykl, double disp,int label){

    std::shared_ptr<Feature> f=Feature::create(cvl::Vector3d(0,0,0));

    std::shared_ptr<Measurement> m=std::make_shared<Measurement>(ykl,disp,frameid, label,f,self);
    // slightly faster than triangulate world, one less dereference of smart ptr...

    m->feature->add(m);
    return m;
}
std::shared_ptr<Measurement> TimePoint::createMeasurement(Vector2d ykl, double disp,int label,
                                                          std::shared_ptr<Measurement>& corr){
std::shared_ptr<Feature> f=corr->feature;
    std::shared_ptr<Measurement> m=std::make_shared<Measurement>(ykl,disp,frameid, label,f,self);
    if(!corr){ cout<<"wierd!=>"<<endl;}    
    f->add(m);
    return m;
}



std::shared_ptr<TimePoint> TimePoint::create( uint frameid, std::shared_ptr<Map> map){
    auto tmp=std::make_shared<TimePoint>(frameid,map);
    tmp->self=tmp;
    return tmp;
}

}

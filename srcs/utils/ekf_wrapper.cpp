#include <sfm/calibration.h>
#include <utils/ekf_wrapper.h>

namespace cvl{
ParamSet* EkfWrapper::getParamSet(){
    return filterparams.getParamSet();
}

SixDVisionFilters::SceneMotion get_scene_motion(PoseD P21)
{

    SixDVisionFilters::SceneMotion sm;
    sm.m_deltaT_f=33.0f/1000.0f; // in seconds probably...
    //sm.m_deltaT_f=1; // to make it comparable, that or divide, might make thresholds off...
    PoseD smp=P21;//P21.inverse();
    for(int i=0;i<9;++i)
        sm.m_rotation[i] =float(smp.getR()[i]);
    sm.m_translationX_f = float(smp.getT()[0]);
    sm.m_translationY_f = float(smp.getT()[1]);
    sm.m_translationZ_f = float(smp.getT()[2]);
    return sm;
}

void EkfWrapper::init(){
    if(inited) return;
    auto c=HirCalibration::common();

    SixDVisionFilters::ImageSize imageSize;
    imageSize.m_width_ui  = c.cols();
    imageSize.m_height_ui = c.rows();
    filters.setImageSize(imageSize);
    SixDVisionFilters::CameraIntrinsics ci; // read this from sd
    {
        ci.m_camFu_f = float(c.fx());
        ci.m_camFv_f = float(c.fy());
        ci.m_camU0_f = float(c.px());
        ci.m_camV0_f = float(c.py());
        ci.m_camBase_f=float(c.baseline());
    }
    filters.setCameraIntrinsics(ci);
    SixDVisionFilters::CameraToWorld c2w;
    {
        for(int i=0;i<9;++i)
            c2w.m_rotation[i]=0;
        c2w.m_rotation[0]=1;
        c2w.m_rotation[4]=1;
        c2w.m_rotation[8]=1;
        c2w.m_translationX_f=0;
        c2w.m_translationY_f=0;
        c2w.m_translationZ_f=0;
    }
    filters.setCameraExtrinsics(c2w);// identity will do... this is vehicle to cam
    filters.setParameters(filterparams.getSystemParameters(),
                          filterparams.getFilterParameters());
    filters.setNumFeatures(30000);
    filters.reset();
}

SixDVisionFilters::Measurement get_6d_measurement(float u, float v, float disp,int state, float varU=0.1f, float varV=0.1f, float varD=0.3f){
    SixDVisionFilters::Measurement m;

    m.m_state_i = state;
    if (disp<0)
        m.m_state_i=-1;

    m.m_d_f=disp;
    m.m_u_f=u;
    m.m_v_f=v;
    m.m_varD_f=varU;
    m.m_varU_f=varV;
    m.m_varV_f=varD;
    return m;
}

double getDim(const cv::Mat1f& dim, cvl::Vector2d ykl){// col,row

    float disp=-100;
    if(ykl[0]>0 && ykl[1]>0)
        if(ykl[0]<dim.cols)
            if(ykl[1]<dim.rows)
                disp=dim(int(std::round(ykl(1))),int(std::round(ykl(0))));
    return disp;

}

std::vector<SixDVisionFilters::Measurement>
get_6d_measurements(CFeaturePool pool, cv::Mat1f dim){
    std::vector<SixDVisionFilters::Measurement> ms;
    ms.resize(pool.getSize(),get_6d_measurement(0,0,0,-1));
    SFeature_t* array=pool.getArray();
    for(uint i=0;i<pool.getSize();++i)
    {
        // so I have the 3d point in current and the

        Vector2d ykl(array[i].u_d,array[i].v_d); // these are col row!
        double disp=getDim(dim,ykl);
        if(disp<=3) continue;

        if(array[i].state==TS_TRACKED )
            ms[i]=get_6d_measurement(array[i].u_d,array[i].v_d,disp,0);
        if(array[i].state==TS_FOUND )
            ms[i]=get_6d_measurement(array[i].u_d,array[i].v_d,disp,1);
    }
    return ms;
}

class Obs{
public:
    Obs(Vector2d ynl,Vector2d ynr,double disp):ynl(ynl),ynr(ynr),disp(disp){}
    Vector2d ynl,ynr;
    double disp;
};

class State6d{
public:
    State6d(SixDVisionFilters::KaFiStateVector kv){
        x=Vector3d(kv.m_x_posX_f,kv.m_x_posY_f,kv.m_x_posZ_f);

        v=Vector3d(kv.m_x_velX_f,kv.m_x_velY_f,kv.m_x_velZ_f);
        if(v.norm()>30) {v.normalize();v*=30;}
        if(x.norm()>500){
            x.normalize();
            x*=500;
            //v*=0; // makes it not a flwo=> nans in the lookat
        }
        v/=1000.0/33.0;
    }
    Vector3d x,v;
    Flow get_flow(){
        return Flow(x, v, Vector3d(0,255,0));
    }
};

std::shared_ptr<FlowField> make_flowfield(
        CFeaturePool pool,
        SixDVisionFilters::KaFiOutputData* out)
{
    // up to 30k
    std::shared_ptr<FlowField> ff=std::make_shared<FlowField>();

    for(uint i=0;i<pool.getSize();++i){
        if(!(out[i].m_age_ui>4 && out[i].m_result_ui==0)) continue;
        Flow f=State6d(out[i].m_state).get_flow();
        if(f.is_normal()){
            if(f.velocity.norm()<1e-1)
            {
                ff->points.points.push_back(f.origin);
                ff->points.colors.push_back(Vector3d(0,255,0));
            }
            else
                ff->flows.push_back(f);
        }
    }
    return ff;
}
std::vector<Vector7d> make_state(
        CFeaturePool pool,
        SixDVisionFilters::KaFiOutputData* out)
{
    std::vector<Vector7d> states;states.reserve(pool.getSize());
    for(uint i=0;i<pool.getSize();++i){

        auto f=State6d(out[i].m_state);
        Vector7d s(0,0,0,0,0,0,0);
        if(f.x.isnormal() &&f.v.isnormal() &&
                (out[i].m_age_ui>0 && out[i].m_result_ui==0)){
            s=Vector7d(f.x[0],f.x[1],f.x[2],f.v[0],f.v[1],f.v[2],1);
        }
        states.push_back(s);
    }
    return states;
}

void EkfWrapper::update(PoseD Pcp, CFeaturePool pool, cv::Mat1f dim){

    filters.setNumFeatures(pool.getSize());

    auto ms=get_6d_measurements(pool,dim);

    filters.setSceneMotion(get_scene_motion(Pcp));
    filters.calculate(&ms[0]);

previous=current;
    current=make_flowfield(pool,filters.getOutputData());
    last_state=make_state(pool,filters.getOutputData());

}

std::vector<Vector7d> EkfWrapper::get_filter_state(){

    return last_state;
}

std::shared_ptr<FlowField> EkfWrapper::get_flowfield(){
    return previous;
}

} // end namespace cvl


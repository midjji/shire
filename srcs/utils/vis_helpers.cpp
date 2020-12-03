#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <vis_helpers.h>
#include <mlib/utils/cvl/pose.h>
#include <mlib/opencv_util/cv.h>
#include <sfm/imo.h>
#include <mlib/utils/string_helpers.h>
#include <experimental/filesystem>
#include <thread>
#include <future>

namespace fs=std::experimental::filesystem;

using std::endl;using std::cout;
namespace cvl{

inline double getDim(const cv::Mat1f& dim, Vector2d ykl){// row,col
    int row=std::round(ykl(0));
    int col=std::round(ykl(1));
    if(row<0|| col<0||dim.rows<row||dim.cols<col) return -100;
    return dim(row,col);
}
cv::Mat3b draw_feature_pool(CFeaturePool pool, cv::Mat3b rgb, cv::Mat1f dim){


    //cout<<"drawRawTracks"<<endl;
    SFeature_t* arr=pool.getArray();
    cv::Mat3b im=rgb.clone();


    for(uint i=0;i<pool.getSize();++i){
        if(arr[i].state!=TS_TRACKED) continue;

        Vector2d ykl(arr[i].row(),arr[i].col());
        Vector2d prevykl(arr[i].last_row(),arr[i].last_col());
        double disp=getDim(dim,ykl);
        if(disp<0) continue;

        mlib::drawArrow(im,ykl,prevykl);
    }
    cout<<"drawRawTracks - done"<<endl;

    return im;

}

/**
 * @brief round_to_xxx
 * @param x
 * @param val
 *
 * This function is intended to be used to reduce the number of bits used to store a number once compressed.
 *  - todo change to round to x bits...
 */
void round_to_xxx(cv::Vec3f& x, double val=4){
    for(int i=0;i<3;++i)
        x[i]=float(std::round(x[i]*val)/val);
}
struct flow_image_t{
    cv::Mat3f image;
    cv::Mat1f count;
    int factor;
    flow_image_t(int rows, int cols, int factor):image(cv::Mat3f(rows/factor,cols/factor,cv::Vec3f(0,0,-1))),
        count(cv::Mat1f(rows/factor,cols/factor,0.0f)),
        factor(factor){}
    void add_flow(cv::Vec3f x, int row, int col){
        if(x[2]<0)x[2]=0;
        int r=row/factor;
        int c=col/factor;
        if(!(r<image.rows)) return;
        if(!(c<image.cols)) return;
        count(r,c)++;
        image(r,c)+=x;
    }
    cv::Mat3f get_final_flow_image(){
        for(int r=0;r<image.rows;++r)
            for(int c=0;c<image.cols;++c){
                cv::Vec3f tmp(0,0,-1); // its either the disparity, the rpe, or -1 for ...
                if(count(r,c)>0)
                    tmp= image(r,c)/count(r,c);
                round_to_xxx(tmp,32);
                image(r,c)=tmp;
            }
        return image;
    }
};


cv::Mat3f make_flow_image(CFeaturePool pool, cv::Mat1f dim, int factor){
    flow_image_t tmp(dim.rows,dim.cols,factor);
    //cout<<"drawRawTracks"<<endl;
    SFeature_t* arr=pool.getArray();
    for(uint i=0;i<pool.getSize();++i){
        if(arr[i].state!=TS_TRACKED) continue;

        Vector2d ykl(arr[i].u_d,arr[i].v_d);
        Vector2d prevykl(arr[i].lastU_d,arr[i].lastV_d);
        double disp=getDim(dim,ykl);
        if(disp<0) continue;
        auto current=ykl;
        auto previous=prevykl;
        Vector2d d= current - previous;
        tmp.add_flow(cv::Vec3f(float(d[1]),float(d[0]),float(disp)),int(std::round(previous[1])),int(std::round(previous[0]))); // in row, col order
    }
    cout<<"drawRawTracks - done"<<endl;
    return tmp.get_final_flow_image();
}
cv::Mat3f make_flow_image(std::vector<std::shared_ptr<Measurement>>& ms, cv::Mat1f dim, int factor){
    ms=not_nullptr_and_has_previous_filter(ms);

    flow_image_t tmp(dim.rows, dim.cols,factor); // delta x, delta y, from previous image to current
    for(auto& m:ms)
    {
        if(m->feature->size()<3) continue;
        double err=m->feature->error();
        if(err>3) continue;
        auto current=m->yl();
        auto previous=m->getPrevious()->yl();
        Vector2d d= current - previous;
        tmp.add_flow(cv::Vec3f(float(d[1]),float(d[0]),float(err)), int(std::round(previous[1])),int(std::round(previous[0]))); // in row, col order
    }
    return tmp.get_final_flow_image();
}
void create_dir2path(fs::path path){
    if(path.filename()!="" )
        path=path.remove_filename();
    fs::create_directories(path);
}
void safe_save_image(cv::Mat im, std::string path){
    create_dir2path(path);
    fs::path pth(path);
    std::string filename=pth.filename();
    std::string dir=pth.remove_filename().string();
    std::string tmp_path=dir+"tmp_"+filename;
    cv::imwrite(tmp_path,im);
    fs::rename(tmp_path,path);
}


void make_flow_dataset(CFeaturePool pool,
                       std::vector<std::shared_ptr<Measurement>> ms,
                       cv::Mat1f dim,
                       std::string basepath,
                       int frameid){

    if(basepath.size()>0)
        if(basepath.back()!='/')
            basepath.push_back('/');

    std::vector<std::pair<cv::Mat3f,std::string>> imgs;

    for(int i=3;i<6;++i)
    {
        imgs.push_back(std::make_pair(make_flow_image(pool,dim,i), basepath+"raw_flow"+str(i)+"/"      + mlib::toZstring(frameid)+".exr"));
        imgs.push_back(std::make_pair(make_flow_image(ms,dim,i),   basepath+"verified_flow"+str(i)+"/" + mlib::toZstring(frameid)+".exr"));
    }

    std::vector<std::thread> thrs;thrs.reserve(imgs.size());
    for(auto imp:imgs)
        thrs.push_back(std::thread([](auto p){safe_save_image(p.first,p.second);}, imp));
    for(auto& thr:thrs)
        thr.join();

}

void draw_box(cv::Mat& im, BoundingBox box, bool gt=true){
    auto color=mlib::Color::nrlarge(box.id).toScalar<cv::Scalar>();
    int thickness=8;
    if(!gt){
        color=mlib::Color::white().toScalar<cv::Scalar>();
        thickness=4;
    }
    int r0=int(box.row_start);
    int c0=int(box.col_start);
    int r1=int(box.row_end);
    int c1=int(box.col_end);
    cv::Point2i a(c0,r0);
    cv::Point2i b(c0,r1);
    cv::Point2i c(c1,r0);
    cv::Point2i d(c1,r1);
    cv::line(im,a,b,color,thickness);
    cv::line(im,a,c,color,thickness);
    cv::line(im,d,b,color,thickness);
    cv::line(im,d,c,color,thickness);
}
cv::Mat3b draw_world_tracks(
        const std::vector<std::shared_ptr<Measurement>>& ms,
        cv::Mat3b rgb,
        double threshold){
    cout<<"drawing tracks"<<endl;
    //pcv->setPointCloud(getTrajectory());
    // also add the camera centers for the imos?!



    int frame_id=-1;
    std::set<std::shared_ptr<PoseImo>> imos;


    for(const std::shared_ptr<Measurement>& a:ms){
        if(!a) continue;

        auto b=a->getPrevious();
        if(!b) continue;
        // only draw older tracks
        if(a->feature->size()<2) continue; // might wanna change this to some other size
        cvl::Vector2d l0=a->yl();
        cvl::Vector2d l1=b->yl();




        if(!a->feature->has_imo()){
            if(!(a->feature->error()<threshold)) continue;
            mlib::drawArrow(rgb,l0,l1,mlib::Color::nr(0),2);
        }else{
            mlib::drawArrow(rgb,l0,l1,a->feature->get_imo()->get_color(),4);
            imos.insert(a->feature->get_imo());
            frame_id=a->frameid;
        }
    }

    for(std::shared_ptr<PoseImo> imo:imos){
        draw_box(rgb,imo->get_imo_bounding_box(frame_id,true),true);
        draw_box(rgb,imo->get_imo_bounding_box(frame_id,false),false);
    }

    cout<<"drawing tracks done"<<endl;
    return rgb;
}
cv::Mat3b draw_candidates(const std::vector<std::shared_ptr<Measurement>>& ms, cv::Mat3b rgb, double threshold){


    for(auto& m:ms)
    {
        if(!m) continue;
        if(!m->getPrevious()) continue;
        if(m->feature->get_imo()) continue; // if I have an imo, continue.
        double err=m->feature->error();

        // draw outliers not on an imo
        if(err>threshold && m->label>0)
        {
            auto b=m->getPrevious();
            PoseD Pwc = m->get_pose().inverse();
            PoseD Pwp = m->getPrevious()->get_pose().inverse();
            Vector3d x=Pwc*m->triangulate();
            auto p=m->getPrevious();
            Vector3d px=Pwp*p->triangulate();
            // approximate velocity for the track in m/s
            // the difference must be atleast 1.5 in disparity?
            // hmm in real velocities...
            double v=(x-px).norm()*30*3.6; // delta time is what? 20 fps, 30? lets say 30
            if(v>255)v=255;


            //cout<<m->obs()<< " "<<(m->obs()-p->obs()).norm()<<" "<<x<<" "<<px<<endl;

            mlib::Color col=mlib::Color::codeDepthRedToDarkGreen(v,0,1).fliprb();
            col=mlib::Color(125,0,v);
            mlib::drawArrow(rgb,b->yl(),m->yl(),col, 2);
        }
    }
    return rgb;
}
}// end namepace cvl


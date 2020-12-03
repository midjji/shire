#include <mlib/utils/mlog/log.h>
#include <mlib/utils/cvl/triangulate.h>

#include <daimler/sample.h>
#include <kitti/odometry/kitti.h>

#include <sfm/calibration.h>
#include <sfm/sample.h>

using std::cout;
using std::endl;
namespace cvl {

cv::Mat3b convertb2rgb8(cv::Mat1b img){
    cv::Mat3b rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c)
            rgb(r,c)=cv::Vec3b(img(r,c),img(r,c),img(r,c));
    return rgb;
}
cv::Mat3b convertw2rgb8(cv::Mat1w img){
    cv::Mat3b rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            uint16_t tmp=img(r,c)/16;
            if(tmp>255)tmp=255;
            rgb(r,c)=cv::Vec3b(tmp,tmp,tmp);
        }
    return rgb;
}
cv::Mat1b convertw2gray8(cv::Mat1w img){
    cv::Mat1b gray(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            uint16_t tmp=img(r,c)/16;
            if(tmp>255)tmp=255;
            gray(r,c)=uint8_t(tmp);
        }
    return gray;
}
cv::Mat3b convertf2rgb8(cv::Mat1f img){
    cv::Mat3b rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            float f=2.5f*img(r,c);
            if(f<0)
                f=0;
            rgb(r,c)=cv::Vec3b(uint8_t(f),uint8_t(f),uint8_t(f));
        }
    return rgb;
}

float HirSample::getDim(double row_, double col_){ // row,col
    int row=int(std::round(row_));
    int col=int(std::round(col_));

    if(row<0) return -100.0f;
    if(col<0) return -200.0f;
    if(int(rows())<row) return -300.0f;
    if(int(cols())<col) return -400.0f;
    if(std::isnan(row+col)) return -500.0f;
    return disparity(row,col);
}
float HirSample::getDim(Vector2d rowcol){return getDim(rowcol[0],rowcol[1]);}


Vector3d HirSample::get_3d_point(double row, double col)
{
    double disp=getDim(row,col);
    if(disp<0) disp=0;
    return HirCalibration::common().triangulate_ray(Vector2d(row,col),disp).dehom();
}
bool HirSample::is_car(Vector2d rowcol){
    return is_car(rowcol(0),rowcol(1));
}
bool HirSample::is_car(double row, double col){
    if(!use_labels) return true;

    assert(row>=0);
    assert(col>=0);
    assert(row<labels.rows);
    assert(col<labels.cols);
    return labels(int(std::round(row)),int(std::round(col)));
} // is disparity still offset by one frame? check !

uint HirSample::rows(){return uint(disparity.rows);}
uint HirSample::cols(){return uint(disparity.cols);}

int HirSample::frameid(){return frameid_;}
int HirSample::sequenceid(){return sequenceid_;}

cv::Mat1b HirSample::disparity_image_grey(){
    cv::Mat1b im(rows(), cols());
    for(uint r=0;r<rows();++r)
        for(uint c=0;c<cols();++c){
            float disp=disparity(r,c);
            disp*=2;
            if(disp<0) disp=0;
            if(disp>255)
                disp=255;
            im(r,c)=uint8_t(disp);
        }
    return im;
}
cv::Mat3b HirSample::disparity_image_rgb(){
    cv::Mat3b im(rows(), cols());
    for(uint r=0;r<rows();++r)
        for(uint c=0;c<cols();++c){
            float disp=disparity(r,c);
            disp*=2;
            if(disp<0) disp=0;
            if(disp>255)
                disp=255;
            im(r,c)=cv::Vec3b(1,1,1)*disp;
        }
    return im;
}
cv::Mat3b HirSample::rgb(uint id) // for visualization, new clone
{
    return convertw2rgb8(images.at(id));


}
cv::Mat1b HirSample::gray(uint id) // for visualization, new clone
{
    return convertw2gray8(images.at(id));
}
cv::Mat3b HirSample::show_labels(){
    cv::Mat3b im(rows(), cols());
    for(uint r=0;r<rows();++r)
        for(uint c=0;c<cols();++c){
            float disp=labels(r,c);
            im(r,c)=cv::Vec3b(1,1,1)*disp;
        }
    return im;
}

std::shared_ptr<HirSample> convert_2_hir_sample(std::shared_ptr<DaimlerSample> sd){
    return std::make_shared<HirSample>(sd->images, sd->dim,sd->labels, 8, sd->frameid());
}
std::shared_ptr<HirSample> convert_2_hir_sample(std::shared_ptr<kitti::KittiOdometrySample> sd){
    cv::Mat1b labels(sd->disparity.rows,sd->disparity.cols,1);        
    return std::make_shared<HirSample>(sd->images, sd->disparity,labels, 8, sd->frameid());
}
}


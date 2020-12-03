#include <sstream>
#include <iostream>
#include <experimental/filesystem>

#include "paramHandling/h/paramseteditor_Qt4.h"
#include <shire/shire.h>

#include <mlib/utils/buffered_dataset_stream.h>
#include <kitti/odometry/kitti.h>
#include <sfm/sample.h>
#include <opencv2/highgui.hpp>
#include <sfm/calibration.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/argparser.h>
namespace fs = std::experimental::filesystem;
using std::cout;using std::endl;


cvl::Shire hir;

int kitti_lhir(int seq, int start){

    hir.set_min_creation_count(10);

    cout<<"starting lhir instance!"<<endl;

    // this is as fast as the parallel one assuming its parallelized internally...
    // the test is chocking on disk

    cvl::kitti::KittiDataset dm("/storage/datasets/kitti/odometry/");

    auto s=dm.getSequence(seq);
    dm.sequence=0;
    cvl::Matrix3d K=s.getK();
    double fx=K(0,0);
    double fy=K(1,1);
    double px=K(0,2);
    double py=K(1,2);
    double baseline=s.baseline;
    double rows=s.rows;
    double cols=s.cols;
    hir.init(rows,cols);


    cvl::HirCalibration::set_common_calibration(
                fx,fy,px,py,baseline,rows,cols);
    cvl::HirCalibration::set_common_calibration_pose(cvl::PoseD());

    mlib::Timer timer("Total time: ");
    mlib::sleep_ms(2000);
    std::shared_ptr<cvl::kitti::KittiOdometrySample> sd;


    //int n=0;
    while(true){ // breaks on nullptr

        //if(n++ % 2>0) continue;
        timer.tic();
        sd=dm.get_sample(seq,start++);
        cout<<sd->cols()<<" "<<sd->rows()<<endl;
        cout<<rows<<" "<<cols<<endl;
        cout<<"read kittisample"<<endl;
        if(!sd) continue;
        std::shared_ptr<cvl::HirSample> sh=cvl::convert_2_hir_sample(sd);
        hir(sh);
        timer.toc();
        cout<<timer<<endl;
        //cv::waitKey(0);
    }
    return 0;
}


int main(int argc, char** argv){
    mlib::ArgParser args;
    args.add_parameter("sequence","which sequence to use", "0");
    args.add_parameter("index","which index to start at", "0");



    QApplication                  myApp(argc,argv);
    CParameterSetEditorQt4        editor;
    editor.addRootParameterSet( hir.getParamSet() );
    cout<<"stargin ip config loop"<<endl;
    std::string parampath="/home/mikael/co/imo/parameters/kitti_imo.dat";

    if(!fs::exists(parampath)){
        cout<<"parameter path not found... "<<endl;
        exit(1);
    }
    editor.load(parampath);

    std::thread thr=std::thread(kitti_lhir, args.param_double(),args.param_double()); // must wait untill ip_config has started!
    mlib::sleep(1);
    editor.show();
    myApp.exec();
    thr.join();
    return 0;
}


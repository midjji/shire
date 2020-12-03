#include <sstream>
#include <iostream>
#include "paramHandling/h/paramseteditor_Qt4.h"
#include <shire/shire.h>

#include <mlib/utils/buffered_dataset_stream.h>
#include <daimler/dataset.h>
#include <sfm/sample.h>
#include <opencv2/highgui.hpp>
#include <sfm/calibration.h>
#include <mlib/utils/string_helpers.h>
#include <experimental/filesystem>
#include <mlib/utils/argparser.h>

using std::cout;using std::endl;
namespace fs = std::experimental::filesystem;
cvl::Shire hir;

void daimler_lhir(int start, std::string dataset_path){

    cout<<"starting lhir instance!"<<start<<endl;

    // this is as fast as th#include <mlib/utils/argparser.h>e parallel one assuming its parallelized internally...
    // the test is chocking on disk
    cvl::BufferedStream<cvl::DaimlerDataset> dm =
            cvl::BufferedStream<
            cvl::DaimlerDataset>(start,dataset_path);



    mlib::Timer timer("Total time: ");
    mlib::sleep_ms(2000);
    std::shared_ptr<cvl::DaimlerSample> sd;
    //int n=0;
    while(true)
    { // breaks on nullptr

        //if(n++ % 2>0) continue;
        timer.tic();
        sd=dm.next();
        cout<<"read daimlersample"<<endl;
        if(!sd) continue;
        std::shared_ptr<cvl::HirSample> sh=cvl::convert_2_hir_sample(sd);
        hir(sh);
        timer.toc();
        cout<<timer<<endl;
        //cv::waitKey(0);
    }
}

int main(int argc, char** argv){
    mlib::ArgParser args;
    args.add_parameter("index","which index to start at", "0");
    args.add_parameter("retrack","","true");
    args.add_parameter("labels","","true");
    args.add_parameter("reproject_threshold","","3");
    args.add_parameter("disambigious_threshold","","4");
    args.add_parameter("candidate_threshold","","5");
    args.add_parameter("min_create_count","","20");
    args.add_parameter("draw_increment","","1");
    args.add_parameter("save_verified_depths","","true");
    args.add_option("--dataset_path",1,"/storage/datasets/daimler/2020-04-26/08/", "path to the dataset!",false);
    args.parse_args(argc,argv);

    int start=args.param_double();
    hir.retrack_world=hir.retrack_world_redo_pnp=args.param_bool();
    hir.use_labels=args.param_bool();
    hir.reprojection_threshold=args.param_double();
    hir.disambiguation_threshold=args.param_double();
    hir.candidate_threshold=args.param_double();
    hir.set_min_creation_count(args.param_double());
    hir.draw_increment=args.param_double();
    hir.save_verified_depths=args.param_bool();

    std::string dataset_path= args.get_arg("--dataset_path");
    if(!fs::exists(dataset_path+"metadata.txt")){
        cout<<"dataset not found!"<<dataset_path<<endl;
        args.help();
    }

    QApplication                  myApp(argc,argv);
    CParameterSetEditorQt4        editor;
    editor.addRootParameterSet( hir.getParamSet() );
    cout<<"stargin ip config loop"<<endl;
    std::string parampath="/home/mikael/co/imo/parameters/imo_retrack.dat";
    if(!fs::exists(parampath))
    {
        cout<<"parameter path not found... "<<endl;
        exit(1);
    }
    editor.load(parampath);

    std::thread thr=std::thread(daimler_lhir, start, dataset_path); // must wait untill ip_config has started!
    mlib::sleep(1);
    editor.show();
    myApp.exec();
    thr.join();
    return 0;
}


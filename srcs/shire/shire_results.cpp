#include <iostream>

#include <shire/shire.h>
#include <mlib/utils/string_helpers.h>
#include <experimental/filesystem>
#include <daimler/database.h>
namespace fs=std::experimental::filesystem;
using std::cout;using std::endl;
namespace cvl{


void Shire::save_results(int result_nr)
{
    cout<<"saving results!"<<endl;
    // A run result gives
    // uid, frameid, ego_pose, // frameid is unique
    // uid, frameid, imo_id, pose(first is origin, this is P_{c,imo}), xm, bb
    // lets run the system for x frames then save everything at that point?
    // every thousand frames?
    std::string path=get_output_directory()+"results/" +
            mlib::toZstring(result_nr,4)+".db.sqlite";


    fs::create_directories(fs::path(path).remove_filename());

    mtable::result_db_type storage =
            sqlite_orm::make_storage(
                path,
                sqlite_orm::make_table(
                    "imo_results",
                    sqlite_orm::make_column("uid", &Imoresrow::uid,
                                            sqlite_orm::autoincrement(),
                                            sqlite_orm::primary_key()),
                    sqlite_orm::make_column("frame_id", &Imoresrow::frame_id),
                    sqlite_orm::make_column("imo_id", &Imoresrow::imo_id),
                    sqlite_orm::make_column("pose0", &Imoresrow::pose0),
                    sqlite_orm::make_column("pose1", &Imoresrow::pose1),
                    sqlite_orm::make_column("pose2", &Imoresrow::pose2),
                    sqlite_orm::make_column("pose3", &Imoresrow::pose3),
                    sqlite_orm::make_column("pose4", &Imoresrow::pose4),
                    sqlite_orm::make_column("pose5", &Imoresrow::pose5),
                    sqlite_orm::make_column("pose6", &Imoresrow::pose6),

                    sqlite_orm::make_column("fxm_x", &Imoresrow::fxm_x),
                    sqlite_orm::make_column("fxm_y", &Imoresrow::fxm_y),
                    sqlite_orm::make_column("fxm_z", &Imoresrow::fxm_z),

                    sqlite_orm::make_column("xm_x", &Imoresrow::xm_x),
                    sqlite_orm::make_column("xm_y", &Imoresrow::xm_y),
                    sqlite_orm::make_column("xm_z", &Imoresrow::xm_z),


                    sqlite_orm::make_column("row_start", &Imoresrow::row_start),
                    sqlite_orm::make_column("col_start", &Imoresrow::col_start),
                    sqlite_orm::make_column("row_end", &Imoresrow::row_end),
                    sqlite_orm::make_column("col_end", &Imoresrow::col_end)
                    ),
                sqlite_orm::make_table(
                    "egomotion",
                    sqlite_orm::make_column("uid", &Egomotionrow::uid, sqlite_orm::autoincrement(), sqlite_orm::primary_key()),
                    sqlite_orm::make_column("frame_id", &Egomotionrow::frame_id),
                    sqlite_orm::make_column("pose0", &Egomotionrow::pose0),
                    sqlite_orm::make_column("pose1", &Egomotionrow::pose1),
                    sqlite_orm::make_column("pose2", &Egomotionrow::pose2),
                    sqlite_orm::make_column("pose3", &Egomotionrow::pose3),
                    sqlite_orm::make_column("pose4", &Egomotionrow::pose4),
                    sqlite_orm::make_column("pose5", &Egomotionrow::pose5),
                    sqlite_orm::make_column("pose6", &Egomotionrow::pose6)),
                sqlite_orm::make_table(
                    "timing",
                    sqlite_orm::make_column("uid", &Timingrow::uid,
                                            sqlite_orm::autoincrement(),
                                            sqlite_orm::primary_key()),
                    sqlite_orm::make_column("frame_id", &Timingrow::frame_id),
                    sqlite_orm::make_column("time_ns", &Timingrow::time_ns)),
                sqlite_orm::make_table(
                    "velocity",
                    sqlite_orm::make_column("uid", &Vdata::uid,
                                            sqlite_orm::autoincrement(),
                                            sqlite_orm::primary_key()),
                    sqlite_orm::make_column("frameid", &Vdata::frameid),
                    sqlite_orm::make_column("imoid", &Vdata::imoid),
                    sqlite_orm::make_column("imo_vx", &Vdata::imo_vx),
                    sqlite_orm::make_column("imo_vy", &Vdata::imo_vy),
                    sqlite_orm::make_column("imo_vz", &Vdata::imo_vz),
                    sqlite_orm::make_column("ekf_vx", &Vdata::ekf_vx),
                    sqlite_orm::make_column("ekf_vy", &Vdata::ekf_vy),
                    sqlite_orm::make_column("ekf_vz", &Vdata::ekf_vz))
                );

    storage.sync_schema();
    int start_frame_id=100000;
    for(auto [frame_id, pose]:map->get_poses()){
        storage.insert(Egomotionrow(-1, frame_id, pose));
        if(frame_id<start_frame_id)
            start_frame_id=frame_id;
    }


    cout<<"imos found: "<<map->getImos(false).size()<<endl;
    for(auto& imo:map->getImos(false)){
        for(Imoresrow r:imo->get_imo_res().resrows())
            storage.insert(r);

    }
    std::vector<mlib::Time> times=timers.make_or_get("total_shire_timer").getTimes();
    for(mlib::Time time:times){
        Timingrow tr(start_frame_id,start_frame_id+1,time.ns);
        start_frame_id++;
        storage.insert(tr);
    }


    for(auto vd:vdatas)
        storage.insert(vd);

    vdatas.clear(); // to avoid double storing the data
    cout<<"saving results done!"<<endl;

}
}

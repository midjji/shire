#pragma once
/* ********************************* FILE ************************************/
/** \file    feature.h
 *
 * \brief    This header contains a imo capable feature
 *
 *
 * \remark
 *
 *
 * \author   Mikael Persson
 * \date     2017-01-01
 * \note BSD licence
 *
 *
 ******************************************************************************/
#include <memory>
#include <vector>
#include <mlib/utils/cvl/pose.h>
namespace cvl{
class Map;
class Measurement;
class TimePoint{
public:

    std::shared_ptr<Measurement> createMeasurement(Vector2d ynl,
                                                   double disp,
                                                   int label,
                                                   std::shared_ptr<Measurement>& corr);
    std::shared_ptr<Measurement> createMeasurement(Vector2d ynl,
                                                   double disp,
                                                   int label);
    static std::shared_ptr<TimePoint> create(uint frameid, std::shared_ptr<Map> map);

    std::shared_ptr<PoseD> Pnw=std::make_shared<PoseD>();
    PoseD pose(){return *Pnw;}

    uint frameid; // frameid is the global frame id. ie for the full dataset, not the subset
    std::weak_ptr<Map> map;
    std::weak_ptr<TimePoint> self;
    TimePoint(uint frameid, std::shared_ptr<Map> map):frameid(frameid),map(map){}

    std::vector<std::shared_ptr<Measurement>> ms;


private:    


};
} // end namespace cvl

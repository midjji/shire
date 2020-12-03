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
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/pose.h>

namespace cvl{
class Measurement;
class PoseImo;
class Feature{
public:

    static std::shared_ptr<Feature> create(const Vector3d& x);
    void add(std::shared_ptr<Measurement>& m);
    std::vector<std::shared_ptr<Measurement> > getMeasurements();
    std::vector<std::shared_ptr<Measurement>> get_last_n(uint n) const;
    // x_world_last ms - x_world_second2last_ms
    Vector3d world_velocity();

    void clear();
    // creates a new track with the new features in it
    std::shared_ptr<Feature> split_track();
    void split_long_track();
    uint size();



    void refine(bool reinitialize);
    std::vector<double> errors(const std::shared_ptr<PoseImo>& imo) const;
    std::vector<double> errors()  const;

    double error() const; // the thing we test thresholds against!
    double error(const std::shared_ptr<PoseImo>& imo)  const; // the thing we test thresholds against!



    Vector3d X;
    std::vector<std::weak_ptr<Measurement>> ms;

    std::shared_ptr<PoseImo> get_imo();
    bool has_imo() const;
    void set_imo(std::shared_ptr<PoseImo>& imo);
    void clear_imo();

    // nullptr if not present!
    std::shared_ptr<Measurement> get_measurement_in_frame(uint frame_id);


    //Vector3d V;
    std::shared_ptr<Feature> get_self();
    int last_ms_frameid() const;
    int imo_age();
private:

    std::shared_ptr<PoseImo> imo=nullptr;
    std::weak_ptr<Feature> self;
    static uint max_num;

};
typedef std::shared_ptr<Feature> sFeature;
} // end namespace cvl

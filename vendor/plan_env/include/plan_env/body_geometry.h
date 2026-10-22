#pragma once
#include <Eigen/Core>
#include <cmath>
#include <vector>

namespace scan345 {
struct BodyPart {
  int kind;
  Eigen::Vector3d center, half;
};
inline bool decodeBody(const std::vector<double>& data, std::vector<BodyPart>& out) {
  if(data.empty() || data.size()%7 || data.size()>7*80)return false;
  std::vector<BodyPart> parts;
  for(size_t i=0;i<data.size();i+=7) {
    for(size_t j=i;j<i+7;++j)if(!std::isfinite(data[j]))return false;
    if(data[i]!=0. && data[i]!=1.)return false;
    BodyPart p{int(data[i]),{data[i+1],data[i+2],data[i+3]},
                           {data[i+4],data[i+5],data[i+6]}};
    if(std::abs(p.center.x())>1.5 || std::abs(p.center.y())>1.5 ||
       p.center.z()<-.5 || p.center.z()>2.5 ||
       (p.half.array()<=0).any() || (p.half.array()>1.3).any())return false;
    if(p.kind==1 && (p.half.array()-p.half.x()).abs().maxCoeff()>1e-9)return false;
    parts.push_back(p);
  }
  out=parts;return true;
}
inline Eigen::Matrix3d yawRotation(double yaw) {
  Eigen::Matrix3d r;
  r<<std::cos(yaw),-std::sin(yaw),0,std::sin(yaw),std::cos(yaw),0,0,0,1;
  return r;
}
inline bool intersectsVoxel(const BodyPart& part,const Eigen::Vector3d& point_in_base,
                            double yaw,double resolution) {
  const Eigen::Vector3d delta=(point_in_base-part.center).cwiseAbs();
  // Occupancy represents a whole voxel, not a point at its centre.
  if(part.kind==1)return delta.norm()<=part.half.x()+std::sqrt(3.)*.5*resolution;
  const double xy=(std::abs(std::cos(yaw))+std::abs(std::sin(yaw)))*.5*resolution;
  return (delta.array()<=(part.half+Eigen::Vector3d(xy,xy,.5*resolution)).array()).all();
}
}  // namespace scan345

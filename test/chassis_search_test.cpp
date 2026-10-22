#include "chassis_search.h"
#include <iostream>
#include <stdexcept>
using namespace chassis;
static void check(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
static void valid(const Result& r,const Free& free) {
  check(r.error.empty(),"search failed");
  check(r.poses.size()==r.edges.size()+1,"pose/edge count");
  for(size_t i=0;i<r.edges.size();++i) {
    check(r.edges[i].v>=0 && r.edges[i].v<=.15 && std::abs(r.edges[i].w)<=.30,"kinematic limits");
    // Independent high-resolution interpolation checks every returned motion.
    int n=std::max(1,int(r.edges[i].t/.01)+1);
    for(int j=0;j<=n;++j)check(free(advance(r.poses[i],r.edges[i],r.edges[i].t*j/n)),"swept collision");
  }
}
int main() {
  Free open=[](const Pose&){return true;};
  auto straight=search({0,0,0},{1,0,0},false,open);valid(straight,open);
  check(straight.final && std::hypot(straight.poses.back().x-1,straight.poses.back().y)<1e-6,"exact endpoint");
  for(int sign:{1,-1}) {
    // The elongated chassis is clear while stationary. An obstacle beside its
    // front corner blocks the direct turn toward the goal, but a detour exists.
    Free near=[sign](const Pose& p) {
      double dx=.40-p.x,dy=sign*.27-p.y;
      double x=std::cos(p.yaw)*dx+std::sin(p.yaw)*dy;
      double y=-std::sin(p.yaw)*dx+std::cos(p.yaw)*dy;
      double qx=std::max(-.25-x,x-.45),qy=std::abs(y)-.20;
      return std::hypot(std::max(0.,qx),std::max(0.,qy))>.035;
    };
    check(near({0,0,0}),"initial pose must be free");
    check(!sweep({0,0,0},motion(0,sign*.55),near),"direct initial turn should collide");
    auto detour=search({0,0,0},{.8,sign*.5,sign*.4},true,near,[](){return false;},5.);
    std::cout<<"detour "<<sign<<" expanded="<<detour.expanded<<" error="<<detour.error<<"\n";
    valid(detour,near);check(detour.final,"detour final");
    check(std::hypot(detour.poses.back().x-.8,detour.poses.back().y-sign*.5)<1e-6,"goal not truncated");
    check(std::abs(wrap(detour.poses.back().yaw-sign*.4))<1e-6,"final yaw");
  }
  auto far=search({0,0,1.2},{6,0,0},false,open);valid(far,open);
  check(!far.final && far.poses.back().x>.8,"far target rolling segment");
  auto blocked=search({0,0,0},{1,0,0},false,[](const Pose&){return false;});
  check(blocked.error=="start_occupied","occupied start explanation");
  auto cancel=search({0,0,0},{1,0,0},false,open,[](){return true;});
  check(cancel.error=="cancelled","cancel");
  for(int i=0;i<8;++i)valid(search({-.3,-.2,2.2},{.5,-.4,0},false,open),open);
  std::cout<<"CHASSIS_SEARCH_TEST_PASS\n";
}

#include "starpath_3d_navigation/approach_geometry.hpp"
#include <cassert>
#include <iostream>
using namespace scan345;
int main(){
 std::vector<Point> route;for(int i=0;i<=125;++i)route.push_back({i*.02,0});
 for(double angle:{0.,pi/6,pi/3,pi/2,-pi/2}){
  auto c=make_curve({0,0},0,.30,0,{2.5,0},angle,.45,route,[](double,double,double){return true;});
  assert(!c.empty());assert(norm(Point{c.q.back().x,c.q.back().y}-Point{2.5,0})<1e-7);
  assert(std::abs(wrap(c.q.back().yaw-angle))<1e-7);assert(std::abs(c.q.front().v-.3)<.026);
  assert(c.q.back().v==0 && std::abs(c.q.back().w)<1e-7);
  for(size_t i=1;i<c.q.size();++i){auto&a=c.q[i-1];auto&b=c.q[i];double dt=b.t-a.t;
   assert(dt>0&&b.v<=.450001&&std::abs(b.w)<=.850001);
   assert(std::abs(b.v-a.v)/dt<=.401);assert(std::abs(b.w-a.w)/dt<=1.651);
  }
  auto q=c.at(c.q.back().t*.5);assert(std::abs(c.project(q.x,q.y)-q.t)<.02);
  std::cout<<"CURVE angle="<<angle<<" length="<<c.q.back().s<<" duration="<<c.q.back().t<<std::endl;
 }
 auto blocked=make_curve({0,0},0,.30,0,{2.5,0},pi/2,.45,route,[](double x,double,double){return x<1.;});assert(blocked.empty());
 assert(make_curve({0,0},0,.3,0,{.1,0},pi/2,.45,route,[](double,double,double){return true;}).empty());
 auto short_turn=make_curve({0,0},0,0,0,{1.5,0},pi/2,.45,route,[](double,double,double){return true;});assert(!short_turn.empty());
 ArrivalHold hold;assert(!hold.hold(.1,.01));assert(!hold.hold(.01,.1));assert(hold.hold(.044,.044));assert(!hold.hold(.051,.03));
 std::cout<<"ALL_GEOMETRY_TESTS_PASSED"<<std::endl;
}

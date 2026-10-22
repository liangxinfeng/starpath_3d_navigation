#include <plan_env/body_geometry.h>
#include <cassert>
#include <iostream>
using namespace scan345;
int main() {
  std::vector<BodyPart> parts;
  assert(decodeBody({0,.02,0,.22,.36,.29,.18,1,.45,.2,1.05,.15,.15,.15},parts));
  assert(!decodeBody({0,0},parts));
  assert(!decodeBody({1,0,0,1,-.1,.1,.1},parts));
  assert(!decodeBody({1,0,0,1,.1,.2,.1},parts));
  assert(!intersectsVoxel(parts[0],{0,.40,.2},0,.1));
  assert(intersectsVoxel(parts[0],{.37,.28,.2},0,.1));
  assert(!intersectsVoxel(parts[1],{0,.624,.92},0,.1));
  assert(!intersectsVoxel(parts[1],{.45,.2,2.5},0,.1));
  assert(intersectsVoxel(parts[1],{.45,.2,1.05},0,.1));
  const Eigen::Vector3d p=yawRotation(M_PI/4)*parts[1].center;
  assert(!intersectsVoxel(parts[1],p,0,.01));
  assert(!intersectsVoxel(parts[1],yawRotation(M_PI/2).transpose()*p,M_PI/2,.01));
  assert(intersectsVoxel(parts[1],yawRotation(M_PI/4).transpose()*p,M_PI/4,.01));
  // A voxel whose centre is outside a box but whose edge touches it is occupied.
  assert(intersectsVoxel(parts[0],{.42,0,.22},0,.1));
  std::cout<<"BODY_GEOMETRY_PASS: chassis, upper-body, overhead, rotation, voxel edges, input validation\n";
}

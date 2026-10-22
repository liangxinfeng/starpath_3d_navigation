#pragma once
#include "approach_geometry.hpp"
namespace scan345 {
struct ApproachContext {bool active=false;std::vector<Knot>reference;};
extern thread_local const ApproachContext* approach_context;
}

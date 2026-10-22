# Continuous terminal pose trajectory

SCAN-only controller plugin; original control_center_navigation package is unchanged.
Adaptive entry up to 3 m. Collision-checked quintic Bezier connector matches
entry position, heading and curvature and goal position/heading with zero final
curvature. Candidates are time-parameterized with forward/backward acceleration
constraints and curvature/yaw-rate bounds. No angle-based forward speed cap.
MPPI tracks synchronized x/y/yaw/v/w reference; phase is projected from measured
progress and preserved across same-goal replans. Existing cost critic and 3D
watchdog remain active. Infeasible connectors fall back to original path critics
and arrival policy. This is a bounded candidate optimizer, not globally optimal
SE(2) trajectory search. Goal is unchanged (5 cm/0.05 rad, stopped 0.35 s).

## Real-chassis arrival refinement

The 2026-10-22 acceptance trace measured 0.14-0.15 s command-to-feedback delay.
The braking profile now reserves 0.16 s response distance and uses 0.30 m/s²
terminal deceleration. Hold enters at 0.045 m / 0.045 rad and releases above
0.05 m / 0.05 rad; the existing final stopped-and-stable checker stays intact.
Fallback yaw bounds depend on remaining yaw and angular deceleration, never on
remaining translation. All actual proposals still pass MPPI footprint checks
and the runtime full-body guard.
The FollowPath endpoint uses a zero ROS timestamp to request the current
map-to-odometry transform. Previously the goal checker could use a historical
transform while the controller used the current transform, delaying success.

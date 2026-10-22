# 星途三维导航 · StarPath

ROS 2 功能包：`starpath_3d_navigation`。基于三维地图与三维定位，为轮式底盘提供全局规划、局部控制、动态避障和桌面测试界面。

## 当前导航流程

```text
星图三维地图与定位 + 当前雷达扫描
              ↓
地面分割 / 自体过滤 / 底盘高度障碍栅格
              ↓
Smac 全局路径 → MPPI 轨迹控制 → 底盘指令
              ↓
实时 URDF 部件碰撞检查 / 底盘控制权与心跳保护
```

当前默认路线是 Smac＋MPPI；仓库同时保留 SCAN-Planner 适配代码和来源说明。名称中的“三维”指三维地图、定位和部件碰撞检查，底盘仍在地面运动，不表示自由空间飞行规划。

## 功能

- 拖选目标位置和终点朝向、轨迹预览与显式执行。
- 末段位置和航向联合轨迹、到点停稳确认。
- 障碍停车后持续重规划，直到目标完成、取消或故障中止。
- WASDQE 手动控制、实时点云与碰撞轮廓显示。
- 平地模式下，地面下方点不参与下陷障碍判别；正高度障碍保留，未知区域不自动当作自由空间。
- 实时障碍仅采用当前有效扫描，不累积历史残留。

## 构建与启动

已在 Ubuntu 24.04 / ROS 2 Jazzy 验证。克隆到 `~/ros2_ws/src/starpath_3d_navigation`，并先安装 [星图三维建图](https://github.com/liangxinfeng/starmap_3d_mapping)。

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
colcon build --packages-select starmap_3d_mapping starpath_3d_navigation --executor sequential --parallel-workers 1 --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run starpath_3d_navigation nav_ui.py
```

UI 可选择已有完整地图并启动三维定位，然后连接导航。先生成轨迹预览；只有显式执行才下发导航目标。

## 依赖与设备适配

除 Nav2、PCL、Eigen、OpenCV、PyQt5、SciPy 等依赖外，还需要原系统的 `control_center_navigation`（ExactGoalSmacPlanner 插件）、`autolife_s3_support`、`autolife_robot_control_center`、底盘控制权组件和原厂 SDK/URDF。完整声明见 `package.xml`。本仓库不是脱离设备驱动即可控制任意机器人的完整发行版。

本次名称去除了机器人编号，底层已验证的 `/scan345/` 命名空间、硬件话题与设备配置保留兼容。新安装使用新包名和生成接口；不要同时运行旧、新导航实例。底盘尺寸来自实际 URDF，未改成固定大矩形；速度与到点控制参数保持当前验证版本。

固定扇形自体过滤会裁去体积内全部点，部署时必须按设备核对。当前是平地导航配置，不提供下陷/台阶落差保护。现有地图与个人配置不随源码发布。

参见 [原导航接入说明](NATIVE_NAVIGATION.md)、[末段控制](JOINT_APPROACH.md)、[实现历史](docs/IMPLEMENTATION_HISTORY.md)。上游来源见 [NOTICE](NOTICE)、`UPSTREAM_SHA256.json`，许可见 [LICENSE](LICENSE)。

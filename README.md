# Sophia AI Formula ROS 2 Workspace

This repository contains public research code from Sophia University's Control Engineering Laboratory for the Honda AI Formula program.

The workspace brings together sensing, perception, navigation, control, simulation, and vehicle integration used during closed-course autonomous-driving research. It is an experimental research workspace rather than a supported product release.

## Repository map

| Area | Purpose |
| --- | --- |
| `aiformula/sensing` | Camera and sensor integration |
| `aiformula/perception` | Road and lane perception, including YOLOP-oriented workflows |
| `aiformula/navigation` | Lane, trajectory, and navigation experiments |
| `aiformula/control` | Vehicle and trajectory-control components |
| `aiformula/vehicle` | Robot/vehicle interfaces |
| `aiformula/launchers` | ROS 2 launch and shell entry points |
| `aiformula/simulator` | Simulation-side integration |
| `newlaneline` | Lane-line research workspace |
| `pid_ws` | PID-related experiments |
| `matlab bayesian optimization based controller parameter tuning` | Controller-tuning experiments |
| `vehicle dynamic(testing)` | Vehicle-dynamics experiments |

## Typical launch flow

The exact hardware configuration changes across experiments. From a configured robot workspace, the historical entry points are:

```bash
cd workspace/ros2_ws/src/aiformula/launchers/shellscript
./init_sensors.sh
ros2 launch launchers all_nodes.launch.py
```

Lane-perception experiments have also used:

```bash
ros2 launch auto_launch auto_yolop_launch.py
```

Obstacle-avoidance experiments compose the road detector, lane-line publisher, lane-point generator, filter, planner, and trajectory follower. Package and executable names under active research can change; inspect the relevant launch file before using them on hardware.

## Research context

Public 2026 work connected to this platform includes:

- Owen Zi-Wen Zhou, Hongkang Yu, Zhewen Zheng, and Wenjing Cao. **A Structure-Consistent Virtual Lane Data Generation Method for Complementing Real-World Lane Data.** MSCS 2026, Toyama, Japan.
- Owen Zi-Wen Zhou, Wei Zhao, and Wenjing Cao. **Autonomous Driving Control of an AI Formula Robot Based on Visual End-to-End Imitation Learning.** JSAE Forum Yokohama 2026. [Official session](https://www.jsae.or.jp/assoc/event/gakkai/forum/2026YOKOHAMA/prog_26-Y8/)

For an individual, evidence-linked research summary, see [Owen Zi-Wen Zhou's AI Formula research portfolio](https://github.com/Tsubashimo-Nanato/ai-formula-research).

## Team and attribution

This repository aggregates work from multiple students and research cohorts under Prof. Wenjing Cao. Code attribution is preserved through Git history. Research results retain their full published author lists and should not be attributed to the repository owner alone.

Earlier repository documentation credited Zhewen Zheng, Mo Chen, Hongkang Yu, and Wei Zhao as student contributors. Additional contributors and research authors are recorded in commit history and linked publications.

## Public-repository boundary

Do not commit:

- private or licensed datasets;
- credentials, device tokens, or network configuration;
- model weights without redistribution permission;
- personal information or participant records;
- internal snapshots or third-party assets without a compatible license.

Hardware commands can move a physical robot. Validate the selected launch files, keep an emergency stop available, and test at low speed in a controlled area.

## Contact and contributions

For public code questions, open a GitHub issue with the relevant package, platform, and ROS 2 version. Institutional and research inquiries should use official Sophia University channels.

## License

The repository is distributed under the [MIT License](LICENSE). Third-party components, datasets, weights, and assets may have separate terms.

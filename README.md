# Sophia AI Formula ROS 2 Research Workspace

This repository is a student-maintained snapshot of code used for AI Formula research in Sophia University's Control Engineering Laboratory. It is not an official Honda software release.

The workspace combines sensing, perception, navigation, control, simulation, and vehicle-integration packages from multiple research cohorts. Some paths and launch commands are historical; inspect them before use on current hardware.

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
| `newlaneline` | Historical lane-line experiments and generated analysis outputs |
| `pid_ws` | Historical PID and trajectory-following workspace, including generated build artifacts |
| `matlab bayesian optimization based controller parameter tuning` | Controller-tuning experiments |
| `vehicle dynamic(testing)` | Vehicle-dynamics experiments |

## Historical launch flow

The exact hardware configuration changed across experiments. In a configured ROS 2 workspace, the historical entry points included:

```bash
cd aiformula/launchers/shellscript
./init_sensors.sh
ros2 launch launchers all_nodes.launch.py
```

Lane-perception experiments have also used:

```bash
ros2 launch auto_launch auto_yolop_launch.py
```

Obstacle-avoidance experiments compose the road detector, lane-line publisher, lane-point generator, filter, planner, and trajectory follower. Package and executable names under active research can change; inspect the relevant launch file before using them on hardware.

## Research context

Related 2026 work includes:

- Owen Zi-Wen Zhou, Hongkang Yu, Zhewen Zheng, and Wenjing Cao. **A Structure-Consistent Virtual Lane Data Generation Method for Complementing Real-World Lane Data.** MSCS 2026, Toyama, Japan, March 6, 2026. [Official program session 3A7](https://www.gakkai-web.net/sice-ctrl/temporary/program.html#point3A7) · [Sophia University activity report, p. 66](https://fst.sophia.ac.jp/wp/wp-content/themes/sophiafst/pdf/2025%E5%B9%B4%E5%BA%A6%E7%9B%AE%E6%AC%A1_%E6%A9%9F%E8%83%BD%E5%89%B5%E9%80%A0%E7%90%86%E5%B7%A5%E5%AD%A6%E7%A7%91.pdf#page=67)
- Owen Zi-Wen Zhou, Wei Zhao, and Wenjing Cao. **Autonomous Driving Control of an AI Formula Robot Based on Visual End-to-End Imitation Learning.** JSAE Forum Yokohama, May 29, 2026. [JSAE session 26-Y8](https://www.jsae.or.jp/assoc/event/gakkai/forum/2026YOKOHAMA/prog_26-Y8/)

The public JSAE program confirms the corresponding Japanese title, date, time, and presenter. The complete author list above comes from the conference materials.

For source-linked summaries of Owen Zi-Wen Zhou's research, see the [AI Formula research portfolio](https://github.com/Tsubashimo-Nanato/aiformula-research). The full author lists above remain authoritative for the cited work.

## Team and attribution

This repository aggregates work from multiple students and research cohorts under Prof. Wenjing Cao. Its current Git history does not provide complete package-level provenance, so repository ownership must not be treated as authorship of all included code.

Earlier repository documentation credited Zhewen Zheng, Mo Chen, Hongkang Yu, and Wei Zhao as student contributors. Research papers are credited separately with their complete author lists; paper authorship does not by itself establish authorship of every package in this workspace.

## Public-repository boundary

This historical snapshot contains generated artifacts and large research files. Before reusing or redistributing a model weight, dataset, PDF, calibration file, or third-party component, verify its owner and license. Do not add:

- private or licensed datasets;
- credentials, device tokens, or network configuration;
- additional model weights without redistribution permission;
- personal information or participant records;
- internal snapshots or third-party assets without a compatible license.

Hardware commands can move a physical robot. Validate the selected launch files, keep an emergency stop available, and test at low speed in a controlled area.

## Contact and contributions

For public code questions, open a GitHub issue with the relevant package, platform, and ROS 2 version. Institutional and research inquiries should use official Sophia University channels.

## License

The repository's own source is distributed under the [MIT License](LICENSE). Third-party components, datasets, weights, documents, and assets may have separate terms; the root license does not automatically relicense them.

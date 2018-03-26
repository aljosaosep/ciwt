# Combined Image- and World-Space Tracking in Traffic Scenes

This repository contains code for the tracking system as described in
**Combined Image- and World-Space Tracking in Traffic Scenes. ICRA 2017. (https://www.vision.rwth-aachen.de/media/papers/paper_final_compressed.pdf)**

By Aljosa Osep, Wolfgang Mehner, Markus Mathias, Bastian Leibe at Computer Vision Group, RWTH Aachen University

![Alt text](image/cover.png?raw=true "Our method.")

## Demo  Video
[Click here to watch the video](https://youtu.be/TCdgUI5Xmus).

## Prerequisite

In order to run the code, your setup has to meet the following minimum requirements (tested versions in parentheses. Other versions might work, too):

* GCC 4.8.4
  * Eigen (3.x)
  * Boost (1.55 or later)
  * OpenCV (3.2.0 + OpenCV contrib)
  * PCL (1.8.x)

In case these are not installed on your system (eg. you have installed your libs to some weird directory such as /home/DOG/local) you need to set manually OpenCV_DIR, PCL_DIR, EIGEN_INCLUDE_DIRS by editing CMakeCache.txt.

## Install

### Compiling the code using CMake
0.  `mkdir build`
0.  `cmake ..`
0.  `make all`

### Running the tracker
0.  Edit the config `%PROJ_DIR%/data/kitti_sample.cfg`, set all the paths.
0.  Run the tracker eg. `CIWTApp --config %PROJ_DIR%/data/kitti_sample.cfg --start_frame 0 --end_frame 15 --show_visualization_2d --show_visualization_3d`
0.  Find a small sample of KITTI tracking dataset in `%PROJ_DIR%/data/kitti_sample` (left/right camera images, Regionlets detections, calibration files).

## Remarks

* Tracking modes
    * There are two tracking modes, `detection` and `detection_shape` (set via `--tracking_mode`, or set in the config)
    * They perform similarly when evaluating MOTA in image-domain (KITTI eval. protocol), `detection_shape` provides significantly more precise localization in the 3D space while the `detection` mode is faster.

* Data preprocessing
    * The tracker requires disparity maps to run, `detection_shape` additionally requires 3D segments (eg. generic object proposals, shipped with the tracker).
    * When you run the tracker for the first time, both will be computed on-the-fly, which will significantly slow-down the proc. time.

* External libraries
    * The tracker ships the following external modules:
        * **libelas** - disparity estimation (http://www.cvlibs.net/software/libelas/)
        * **libviso2** - egomotion estimation (http://www.cvlibs.net/software/libviso/)

* Etc
    * The tracking framework does not ship a scene-flow estimator (you can get one here https://github.com/vogechri/PRSM)
    * In the paper experiments, we additionally used a scene-flow estimator to obtain velocity estimates of the 3D segments. You can input to the tracker velocity maps via `--flow_map_path`, but it is not necessary. Tracker will work just fine without it.


* Run the tracker in `release` mode (oterwise it will be slow).

If you have any issues or questions about the code, please contact me https://www.vision.rwth-aachen.de/person/13/

## Citing

If you find the tracker useful in your research, please consider citing:

    @inproceedings{Osep17ICRA,
      title={Combined Image- and World-Space Tracking in Traffic Scenes},
      author={O\v{s}ep, Aljo\v{s}a and Mehner, Wolfgang and Mathias, Markus and Leibe, Bastian},
      booktitle={ICRA},
      year={2017}
    }

## License

GNU General Public License (http://www.gnu.org/licenses/gpl.html)

Copyright (c) 2017 Aljosa Osep
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

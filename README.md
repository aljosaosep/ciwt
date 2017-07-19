# Combined Image- and World-Space Tracking in Traffic Scenes

This repository contains code for the tracking system as described in
**Aljosa Osep, Wolfgang Mehner, Markus Mathias, Bastian Leibe. Combined Image- and World-Space Tracking in Traffic Scenes. ICRA 2017. (https://www.vision.rwth-aachen.de/media/papers/paper_final_compressed.pdf)**

## Demo  Video
TODO

## Prerequisite

In order to run the code, your setup has to meet the following minimum requirements (tested versions in parentheses. Other versions might work, too):

* GCC 4.8.4
  * Eigen (3.x)
  * Boost (1.55 or later)
  * OpenCV (3.2.0 + OpenCV contrib)
  * PCL (1.8.x)

## Install

### Compiling the code using CMake
0.  `mkdir build`
0.  `cmake ..`
0.  `make all`

### Running the tracker
0.  `CIWTApp --config %PROJ_DIR%/data/kitti_sample.cfg --start_frame 0 --end_frame 15 --show_visualization_2d --show_visualization_3d

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
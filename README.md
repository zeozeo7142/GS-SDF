# GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction

### ⭐ News
- 2026/02/03: Update [Custom FAST-LIVO2 Datasets](#44-custom-fast-livo2-datasets) for better adaption to general collection settings.
- 2025/08/09: Support colmap-format [Multi-camera datasets](#44-multi-camera-datasets).

## 1. Introduction

![alt text](pics/pipeline.jpg)
A unified LiDAR-visual system achieving geometrically consistent photorealistic rendering and high-granularity surface reconstruction.
We propose a unified LiDAR-visual system that synergizes Gaussian splatting with a neural signed distance field. The accurate LiDAR point clouds enable a trained neural signed distance field to offer a manifold geometry field. This motivates us to offer an SDF-based Gaussian initialization for physically grounded primitive placement and a comprehensive geometric regularization for geometrically consistent rendering and reconstruction.

Our paper is currently undergoing peer review. The code will be released once the paper is accepted.

[Project page](https://jianhengliu.github.io/Projects/GS-SDF/) | [Paper](https://arxiv.org/pdf/2503.10170) | [Video](https://youtu.be/w_l6goZPfcI)

## 2. Related paper

[GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction](https://arxiv.org/pdf/2503.10170)

[FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/pdf/2408.14035)  

If you use GS-SDF for your academic research, please cite the following paper. 
```bibtex
@article{liu2025gssdflidaraugmentedgaussiansplatting,
      title={GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction}, 
      author={Jianheng Liu and Yunfei Wan and Bowen Wang and Chunran Zheng and Jiarong Lin and Fu Zhang},
      journal={arXiv preprint arXiv:2108.10470},
      year={2025},
}
```

## 3. Installation

- Tested on Ubuntu 20.04, cuda 11.8

> The software not relies on ROS, but under ROS noetic installed, the installation should be easier.
> And if real-time visualization is needed, ROS is required and refer to the [Visualization](#3-visualization) section.

```bash
  pip install open3d==0.18.0
  # Libtorch
  wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu118.zip
  apt install zip
  unzip libtorch-cxx11-abi-shared-with-deps-*.zip
  rm *.zip
  echo "export Torch_DIR=$PWD/libtorch/share/cmake/Torch" >> ~/.bashrc # ~/.zshrc if you use zsh
  source ~/.bashrc # .zshrc if you use zsh

  mkdir -p gs_sdf_ws/src
  cd gs_sdf_ws/src
  apt install git libdw-dev
  git clone https://github.com/hku-mars/GS-SDF.git --recursive
  cd ..
```

#### Build with ROS for visualization

```bash
  catkin_make -j8 -DENABLE_ROS=ON
```

#### (Alternative) Build without ROS

```bash
  # Instead of build with catkin_make, you can also build with cmake
  cd gs_sdf_ws/src/GS-SDF
  mkdir build
  cd build
  cmake ..
  make -j8
```


#### Troubleshooting1: Directly patch labeled_partition to work in CUDA 11.8
```bash
cat > /tmp/patch_labeled_partition.py << 'EOF'
import re

# 패치할 파일 목록
files = [
    "/ws/gs_sdf_ws/src/GS-SDF/submodules/gsplat_cpp/submodules/gsplat/gsplat/cuda/csrc/Projection2DGSFused.cu",
    "/ws/gs_sdf_ws/src/GS-SDF/submodules/gsplat_cpp/submodules/gsplat/gsplat/cuda/csrc/Projection2DGSPacked.cu",
    "/ws/gs_sdf_ws/src/GS-SDF/submodules/gsplat_cpp/submodules/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu",
    "/ws/gs_sdf_ws/src/GS-SDF/submodules/gsplat_cpp/submodules/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu",
]

# labeled_partition 대체 헬퍼 함수 (파일 상단에 추가)
helper = '''
// CUDA 11.8 compatibility: labeled_partition replacement
#if __CUDACC_VER_MAJOR__ < 12
namespace cg = cooperative_groups;
template<typename T>
__device__ inline cooperative_groups::coalesced_group labeled_partition_compat(
    cooperative_groups::coalesced_group const& warp, T key) {
    unsigned int mask = __match_any_sync(warp.ballot(1), (unsigned long long)key);
    return cooperative_groups::coalesced_threads();
}
#define LABELED_PARTITION(warp, key) cooperative_groups::coalesced_threads()
#else
#define LABELED_PARTITION(warp, key) cg::labeled_partition(warp, key)
#endif
'''

for fpath in files:
    with open(fpath, 'r') as f:
        content = f.read()

    # 이미 패치됐으면 스킵
    if 'LABELED_PARTITION' in content:
        print(f"Already patched: {fpath}")
        continue

    # labeled_partition 호출을 매크로로 교체
    content = content.replace(
        'cg::labeled_partition(warp, gid)',
        'LABELED_PARTITION(warp, gid)'
    )
    content = content.replace(
        'cg::labeled_partition(warp, cid)',
        'LABELED_PARTITION(warp, cid)'
    )

    # 첫 번째 #include 다음에 헬퍼 삽입
    content = re.sub(
        r'(#include\s+<cooperative_groups\.h>)',
        r'\1' + helper,
        content,
        count=1
    )
    # cooperative_groups.h include가 없으면 첫 #include 앞에 추가
    if 'LABELED_PARTITION' not in content or helper not in content:
        first_include = content.find('#include')
        if first_include != -1:
            content = content[:first_include] + helper + '\n' + content[first_include:]

    with open(fpath, 'w') as f:
        f.write(content)
    print(f"Patched: {fpath}")

print("Done!")
EOF

python3 /tmp/patch_labeled_partition.py
```

#### Troubleshooting2: tiny-cuda-nn, tcnn_binding build
```bash
cat > /ws/gs_sdf_ws/fix_and_build.sh << 'EOF'
#!/bin/bash
set -e

WORKSPACE="/ws/gs_sdf_ws"
FLAGS_MAKE="$WORKSPACE/build/GS-SDF/submodules/tcnn_binding/submodules/tiny-cuda-nn/CMakeFiles/tiny-cuda-nn.dir/flags.make"

echo "=== Step 1: cmake 단계만 실행 ==="
cd "$WORKSPACE"
rm -rf build devel

export TCNN_CUDA_ARCHITECTURES=86
export CPLUS_INCLUDE_PATH=/usr/include/python3.8:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/usr/include/python3.8:$C_INCLUDE_PATH

# cmake만 실행 (make는 실행하지 않음)
cmake /ws/gs_sdf_ws/src \
    -DENABLE_ROS=ON \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_FLAGS="" \
    -DCATKIN_DEVEL_PREFIX=/ws/gs_sdf_ws/devel \
    -DCMAKE_INSTALL_PREFIX=/ws/gs_sdf_ws/install \
    -G "Unix Makefiles" \
    -B /ws/gs_sdf_ws/build

echo ""
echo "=== Step 2: tiny-cuda-nn flags.make 수정 ==="
python3 -c "
import shutil
f = '$FLAGS_MAKE'
shutil.copy2(f, f + '.bak')
lines = open(f).readlines()
new = []
for l in lines:
    if l.startswith('CUDA_FLAGS'):
        l = 'CUDA_FLAGS = --expt-relaxed-constexpr --expt-extended-lambda --extended-lambda -Xcompiler=-Wno-float-conversion -Xcompiler=-fno-strict-aliasing -Xcudafe=--diag_suppress=unrecognized_gcc_pragma --generate-code=arch=compute_86,code=[compute_86,sm_86] -Xcompiler=-fPIC -std=c++14\n'
        print('새 CUDA_FLAGS 적용:', l[:80], '...')
    new.append(l)
open(f, 'w').writelines(new)
print('flags.make 수정 완료')
"

echo ""
echo "=== Step 3: make 실행 ==="
make -j8 -C /ws/gs_sdf_ws/build

echo ""
echo "=== 빌드 완료 ==="
EOF

chmod +x /ws/gs_sdf_ws/fix_and_build.sh
bash /ws/gs_sdf_ws/fix_and_build.sh
```

## 4. Data Preparation

- The processed FAST-LIVO2 Datasets and Replica Extrapolation Datasets are available at [M2Mapping Datasets](https://furtive-lamprey-00b.notion.site/M2Mapping-Datasets-e6318dcd710e4a9d8a4f4b3fbe176764)

### 4.1. Replica

- Download the Replica dataset from [M2Mapping Datasets](https://furtive-lamprey-00b.notion.site/M2Mapping-Datasets-e6318dcd710e4a9d8a4f4b3fbe176764) and unzip it to `src/GS-SDF/data`:
  ```bash
  wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
  # Replica.zip, cull_replica_mesh.zip, and replica_extra_eval.zip are supposed under gs_sdf_ws
  unzip -d src/GS-SDF/data Replica.zip
  unzip -d src/GS-SDF/data/Replica cull_replica_mesh.zip
  unzip -d src/GS-SDF/data replica_extra_eval.zip
  ```
- Arrange the data as follows:
  ```bash
  ├── Replica
  │   ├── cull_replica_mesh
  │   │   ├── *.ply
  │   ├── room2
  │   │   ├── eval
  │   │   │   └── results
  │   │   │   │   └── *.jpg
  │   │   │   │   └── *.png
  │   │   │   └── traj.txt
  │   │   └── results
  │   │   │   └── *.jpg
  │   │   │   └── *.png
  │   │   └── traj.txt
  ```

### 4.2. FAST-LIVO2 Datasets

- Download either Rosbag or Parsered Data in [M2Mapping Datasets](https://furtive-lamprey-00b.notion.site/M2Mapping-Datasets-e6318dcd710e4a9d8a4f4b3fbe176764).
- Arrange the data as follows:

  - For Rosbag:
    ```bash
    ├── data
    │   ├── FAST_LIVO2_Datasets
    │   ├── campus
    │   │   │   ├── fast_livo2_campus.bag
    ```
  - For Parsered Data:
    ```bash
    ├── data
    │   ├── FAST_LIVO2_Datasets
    │   │   ├── campus
    │   │   │   ├── images
    │   │   │   ├── depths
    │   │   │   ├── color_poses.txt
    │   │   │   ├── depth_poses.txt
    ```

### 4.3. Custom FAST-LIVO2 Datasets

- Clone the [modified-FAST-LIVO2](https://github.com/jianhengLiu/FAST-LIVO2) repo; install and run FAST-LIVO2 as the official instruction. The overall pipeline as:
  ```bash
  # 1. open a terminal to start LIVO
  roslaunch fast_livo mapping_avia.launch
  # 2. open another terminal to get ready for bag recording
  rosbag record /aft_mapped_to_init_lidar /aft_mapped_to_init_cam /origin_img/compressed /cloud_registered_body /tf /tf_static /path -O "fast_livo2_YOUR_DOWNLOADED" -b 4096 -O YOUR_BAG_NAME.bag
  # 3. open another terminal to play your downloaded/collected bag
  rosbag play YOUR_DOWNLOADED.bag
  # 4. convert rosbag into colmap format
  python scripts/rosbag_convert/rosbag_to_colmap.py \                       
    --bag_path data/YOUR_BAG_NAME.bag \--image_topic /origin_img/compressed \
    --image_pose_topic /aft_mapped_to_init_cam \
    --point_topic /cloud_registered_body \
    --point_pose_topic /aft_mapped_to_init_lidar \
    --output_dir data/YOUR_BAG_NAME_colmap \
    --fx [fx] --fy [fy] --cx [cx] --cy [cy] \
    --width [width] --height [height] \
    --k1=[k1] --k2=[k2] --p1=[p1] --p2=[p2]
  # 5. run GS-SDF with the converted colmap format data
  rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/colmap/colmap_example.yaml data/YOUR_BAG_NAME_colmap
  ```

### 4.4. Multi-camera datasets

- Following [Colmap-txt-format](https://colmap.github.io/format.html) to prepare the multi-camera datasets as follows:
  ```bash
  ├── data
  │   ├── colmap_dataset
  │   │   ├── cameras.txt
  │   │   ├── images.txt
  │   │   ├── depths.txt
  │   │   ├── images/
  │   │   ├── depths/
  ```
  You can download the multi-camera demo datasets from [M2Mapping Datasets](https://furtive-lamprey-00b.notion.site/M2Mapping-Datasets-e6318dcd710e4a9d8a4f4b3fbe176764):
  ```bash
  rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/colmap/shenzhenbei.yaml src/GS-SDF/data/multi_cam_demo_shenzhenbei_202404041751
  ```

## 5. Run

```bash
    source devel/setup.bash # or setup.zsh

    # Replica
    ./src/GS-SDF/build/neural_mapping_node train src/GS-SDF/config/replica/replica.yaml src/GS-SDF/data/Replica/room2
    # If ROS is installed, you can also run the following command:
    # rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/replica/replica.yaml src/GS-SDF/data/Replica/room2

    # FAST-LIVO2 (ROS installed & ROS bag)
    ./src/GS-SDF/build/neural_mapping_node train src/GS-SDF/config/fast_livo/campus.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/campus/fast_livo2_campus.bag
    # If ROS is installed, you can also run the following command:
    # rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/fast_livo/campus.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/campus/fast_livo2_campus.bag

    # FAST-LIVO2 (Parsered ROS bag format)
    ./src/GS-SDF/build/neural_mapping_node train src/GS-SDF/config/fast_livo/campus.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/campus/color_poses.txt
    # If ROS is installed, you can also run the following command:
    # rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/fast_livo/campus.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/campus/color_poses.txt
```

After running, the training and evaluation results will be saved in the `src/GS-SDF/output` directory.

For afterward visualization/evaluation, you can use the following command:

```bash
    source devel/setup.bash # or setup.zsh
    ./src/GS-SDF/build/neural_mapping_node view src/GS-SDF/output/(your_output_folder)
    # If ROS is installed, you can also run the following command:
    # rosrun neural_mapping neural_mapping_node view src/GS-SDF/output/(your_output_folder)
```

Input `h` + `Enter` to see the help message.

- **Use provided scripts to reproduce the results:**
  ```bash
      cd src/GS-SDF
      sh scripts/baseline.sh
  ```

## 6. Visualization

- Tested on Ubuntu 20.04, cuda 11.8, ROS Noetic
- We use RVIZ for visualization for now. Please install ROS Noetic following the [official guide](http://wiki.ros.org/noetic/Installation/Ubuntu) or refer to the [Docker](#6-docker) 'ROS Installation' section.
- Re-build the packege:

  ```bash
  cd src
  git clone https://github.com/jianhengLiu/rviz_map_plugin.git
  git clone https://github.com/jianhengLiu/rviz_fps_plugin.git  
  sudo apt install ros-noetic-mesh-msgs ros-noetic-rviz-animated-view-controller ros-noetic-hdf5-map-io
  catkin_make -DENABLE_ROS=ON
  ```
- Run the following command to visualize the map in real-time:

  ```bash
  source devel/setup.bash # or setup.zsh
  roslaunch neural_mapping rviz.launch
  ```

  Click the `FPS Motion` button to enable FPS control, and you can use the `W`, `A`, `S`, `D` keys to move around the map. Drag the view to activate and control the view with the mouse.
- For post-training visualization, you can use the following command:

  ```bash
  ./src/GS-SDF/build/neural_mapping_node view src/GS-SDF/output/(your_output_folder)
  # If ROS is installed, you can also run the following command:
  # rosrun neural_mapping neural_mapping_node view src/GS-SDF/output/(your_output_folder)

  roslaunch neural_mapping rviz.launch
  ```

## 7. Docker

- We provide a [enroot](https://github.com/NVIDIA/enroot) docker image for testing.
  ```bash
  # https://github.com/NVIDIA/enroot
  enroot import docker://nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
  enroot create --name gs_sdf ~/nvidia+cuda+11.8.0-cudnn8-devel-ubuntu20.04.sqsh
  # check if create right
  enroot list
  enroot start --root --rw gs_sdf
  # ctrl + d to return

  cd ~
  # ROS Installation
  apt update
  apt install lsb-release
  sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
  apt install curl # if you haven't already installed curl
  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
  apt update
  apt install ros-noetic-desktop-full
  # Dependencies
  apt install ros-noetic-mesh-msgs ros-noetic-rviz-animated-view-controller ros-noetic-hdf5-map-io
  echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

  # Libtorch
  wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu118.zip
  apt install zip
  unzip libtorch-cxx11-abi-shared-with-deps-*.zip
  rm *.zip
  echo "export Torch_DIR=$PWD/libtorch/share/cmake/Torch" >> ~/.bashrc
  source ~/.bashrc

  # upgrad cmake
  wget https://github.com/Kitware/CMake/releases/download/v3.23.0/cmake-3.23.0-linux-x86_64.sh
  bash ./cmake-3.23.0-linux-x86_64.sh --skip-licence --prefix=/usr 
  # opt1: y; opt2: n

  mkdir -p m2mapping_ws/src
  cd m2mapping_ws/src
  apt install git libdw-dev
  git clone https://github.com/hku-mars/GS-SDF.git --recursive
  cd ..
  catkin_make -DENABLE_ROS=ON # if lacking memory try restricting number of cores: catkin_make -j8

  # Image export
  enroot export --output gs_sdf.sqsh gs_sdf
  ```

## 8. Acknowledgement

Thanks for the excellent open-source projects that we rely on:
[gsplat](https://github.com/nerfstudio-project/gsplat), [M2Mapping](https://github.com/hku-mars/M2Mapping), [nerfacc](https://github.com/nerfstudio-project/nerfacc), [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), [kaolin-wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp), [CuMCubes
](https://github.com/lzhnb/CuMCubes)

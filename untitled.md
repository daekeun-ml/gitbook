# MXNet Installation on NVIDIA Jetson Nano

## Problem Statement

AWS 공식 문서에서 설치법이 있지만 재작년 기준의 내용이고, CUDA 9.0으로 pre-compiled된 라이브러리이기 때문에, CUDA 10.0이 기본으로 설치된 Jetson Nano에선 그대로 사용할 수 없음.

다행히\(?\) MXNet 공식 홈페이지와 NVIDIA 포럼에서 1.4.1 기준의 설치 방법을 제시하고 있지만, 현 시점에서는 OpenCV 의존성 이슈로 pre-compiled 라이브러리 대신 직접 패키지를 빌드해야 함.

### References

#### MXNet

* [https://mxnet.apache.org/get\_started/jetson\_setup](https://mxnet.apache.org/get_started/jetson_setup)
* [https://forums.developer.nvidia.com/t/i-was-unable-to-compile-and-install-mxnet-on-the-jetson-nano-is-there-an-official-installation-tutorial/72259\#5326170](https://forums.developer.nvidia.com/t/i-was-unable-to-compile-and-install-mxnet-on-the-jetson-nano-is-there-an-official-installation-tutorial/72259#5326170)
* [https://forums.developer.nvidia.com/t/i-was-unable-to-compile-and-install-mxnet1-5-with-tensorrt-on-the-jetson-nano-is-there-someone-have-compile-it-please-help-me-thank-you/111303\#5426042](https://forums.developer.nvidia.com/t/i-was-unable-to-compile-and-install-mxnet1-5-with-tensorrt-on-the-jetson-nano-is-there-someone-have-compile-it-please-help-me-thank-you/111303#5426042)
* [https://qiita.com/sparkgene/items/425d310c1d6c9158f896](https://qiita.com/sparkgene/items/425d310c1d6c9158f896) \(일본어 페이지이지만, 구글 번역으로 쉽게 해석 가능\)

#### OpenCV

* [https://bluexmas.tistory.com/977](https://bluexmas.tistory.com/977)
* [https://medium.com/@dmccreary/getting-your-camera-working-on-the-nvida-nano-336b9ecfed3a](https://medium.com/@dmccreary/getting-your-camera-working-on-the-nvida-nano-336b9ecfed3a)

## \(Optional\) 삽질기

아래 제안 방법들은 포럼에서 제안된 방법들로 삽질기를 기록해 둠. 곧바로 **Installing OpenCV** 로 넘어가도 무방함.

#### 제안 방법 1: `mxnet-jetson` 패키지 설치

* 실험 결과: mxnet 1.2.1 기준이므로 여전히 동일한 문제 발생

```bash
pip install mxnet-jetson
```

#### 제안 방법 2. MXNet 1.4.1 prebuilt-package 설치

python2 기준

```bash
sudo apt-get install -y git build-essential libatlas-base-dev libopencv-dev graphviz python-pipsudo 
pip install mxnet-1.4.0-cp27-cp27mu-linux_aarch64.whl
```

python3 기준

```bash
sudo apt-get install -y git build-essential libatlas-base-dev libopencv-dev graphviz python3-pipsudo 
pip install mxnet-1.4.0-cp36-cp36m-linux_aarch64.whl
```

* 실험 결과: `libopencv_imgcodecs.so.3.3` 관련 에러 발생. \(python2/3 동일\)
* 이는 pre-compiled 라이브러리가 OpenCV 3.3을 인스톨했다고 가정하고 OpenCV 의존성을 포함시켜 발생하는 문제로, 결국 OpenCV 3.3을 따로 인스톨해야 함.

```python
>>> import mxnet

Pdb)continue
 Traceback (most recent call last):
 File “/usr/lib/python3.6/pdb.py”, line 1667, in main
 pdb._runscript(mainpyfile)
 File “/usr/lib/python3.6/pdb.py”, line 1548, in _runscript
 self.run(statement)
 File “/usr/lib/python3.6/bdb.py”, line 434, in run
 exec(cmd, globals, locals)
 File “<string>”, line 1, in <module>
 File “/home/lycaass/mximport.py”, line 1, in <module>
 import mxnet
 File “/usr/local/lib/python3.6/dist-packages/mxnet/*init*.py”, line 24, in <module>
 from .context import Context, current_context, cpu, gpu, cpu_pinned
 File “/usr/local/lib/python3.6/dist-packages/mxnet/context.py”, line 24, in <module>
 from .base import classproperty, with_metaclass, _MXClassPropertyMetaClass
 File “/usr/local/lib/python3.6/dist-packages/mxnet/base.py”, line 213, in <module>
 _LIB = _load_lib()
 File “/usr/local/lib/python3.6/dist-packages/mxnet/base.py”, line 204, in _load_lib
 lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_LOCAL)
 File “/usr/lib/python3.6/ctypes/*init*.py”, line 348, in *init*
 self._handle = _dlopen(self._name, mode)
 OSError: libopencv_imgcodecs.so.3.3: cannot open shared object file: No such file or directory
```

#### 제안 방법 3: MXNet 1.6.0 prebuilt-package 설치

* 실험 결과: 방법 2와 동일한 문제 발생
* 결국 OpenCV 3.3을 설치하거나, MXNet을 재컴파일해야 함.
* `sudo-apt ilbopencv-dev python-opencv`로 설치하면 버전 충돌 발생! \(`libopencv-dev: 4.1.1, python-opencv: 3.2.0`\)

## Installing OpenCV

아래 설치 방법들은 pip, numpy, scipy, cmake가 이미 설치되었다고 가정함. 특히 scipy 컴파일이 생각보다 오래 걸리기 때문에, 별도로 분리하는 것을 추천

MXNet 컴파일 시 OpenCV를 제외하고 컴파일 가능하나, 이렇게 하면 향후 inference 디버깅에 난항을 겪기에 한꺼번에 설치하는 것을 추천

#### 1\) pkg-config으로 현재 설치된 OpenCV 확인

Jetson Nano의 jetpack 4.3에서 OpenCV 3.3.1이 디폴트로 깔려 있다고 하는데, 본인의 Jetson Nano에서는 인식이 되지 않음.

```bash
pkg-config opencv --modversion
```

#### 2\) 쉘스크립트 다운로드

OPENCV\_GENERATE\_PKGCONFIG를 활성화하는 것을 권장하며, 이는 쉘스크립트에 모두 반영해 놓았음.

```bash
#!/bin/bash
#
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
# Modified by Daekeun Kim
# [Mofidifations]
# - Activated OPENCV_GENERATE_PKGCONFIG
# - Removed the script; sudo apt-get install -y python2.7-dev python3.6-dev python-dev python-numpy python3-numpy
#   (Assume we have installed python and numpy)
# - Removed the script due to version conflicting; sudo apt-get install -y python-opencv python3-opencv
#   (The latest version of python-opencv is 3.2.0, not 3.3.x or 4.x)
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <Install Folder>"
    exit
fi
folder="$1"
user="nvidia"
passwd="nvidia"
echo "** Remove OpenCV3.3 first"
sudo sudo apt-get purge *libopencv*
echo "** Install requirement"
sudo apt-get update
sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get install -y libv4l-dev v4l-utils qv4l2 v4l2ucp
sudo apt-get install -y curl
sudo apt-get update
echo "** Download opencv-4.1.1"
cd $folder
curl -L https://github.com/opencv/opencv/archive/4.1.1.zip -o opencv-4.1.1.zip
curl -L https://github.com/opencv/opencv_contrib/archive/4.1.1.zip -o opencv_contrib-4.1.1.zip
unzip opencv-4.1.1.zip 
unzip opencv_contrib-4.1.1.zip 
cd opencv-4.1.1/
echo "** Building..."
mkdir release
cd release/
cmake -D WITH_CUDA=ON -D CUDA_ARCH_BIN="5.3" -D CUDA_ARCH_PTX="" -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.1/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_opencv_python2=ON -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON ..
make -j3
sudo make install
echo "** Install opencv-4.1.1 successfully"
echo "** Bye :)"
```

[opencv4.sh](https://quip-amazon.com/-/blob/Kfd9AAPMUOB/wRVnc-U_kLpKwhbehRf1YA?name=opencv4.sh)

#### 3\) 쉘스크립트 실행

약 4시간 소요됨. 설치 후에는 재부팅을 수행해야 import가 정상 동작 \(예: import cv2\)

```bash
mkdir opencv411
chmod 755 opencv4.sh
sudo ./opencv4.sh opencv411
```

#### 4\) pkg-config 설정

최신 OpenCV는 `OPENCV_GENERATE_PKGCONFIG` 옵션을 활성화해도 pkg-config를 인식하지 못하는 문제 발생. 이를 간단한 방법으로 해결 가능.

```bash
cd /usr/local/lib/pkgconfig
cp opencv4.pc opencv.pc
pkg-config opencv --modversion

> 4.1.1
```

#### 5\) \(Optional\) 테스트

```text
cd opencv411/opencv-4.1.1/samples/python
python opt_flow.py
```

```
jetson_release
```

#### 6\) \(Optional\) 웹캠 테스트

웹캠이 정상적으로 인식되는지 확인

```bash
ls /dev/video0
```

[https://github.com/JetsonHacksNano/CSI-Camera](https://github.com/JetsonHacksNano/CSI-Camera) 의 코드를 git clone 후 테스트

```bash
git clone https://github.com/JetsonHacksNano/CSI-Camera.git
cd CSI-Camera
gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! nvoverlaysink
```

왼쪽 커넥터에 웹캠 연결 시 `sensor_id = 0`이고, 오른쪽 커넥터에 연결 시 `sensor_id = 1`

OpenCV Haar-cascade Face Detector 테스트; 아래 코드를 복붙하여 `face-detector.py`로 저장 후 `python face-detector.py`로 실행

```python
import cv2
import numpy as np
HAAR_CASCADE_XML_FILE_FACE = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

def faceDetect():
    # Obtain face detection Haar cascade XML files from OpenCV
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_XML_FILE_FACE)

    # Video Capturing class from OpenCV
    video_capture = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        cv2.namedWindow("Face Detection Window", cv2.WINDOW_AUTOSIZE)

        while True:
            return_key, image = video_capture.read()
            if not return_key:
                break

            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)

            # Create rectangle around the face in the image canvas
            for (x_pos, y_pos, width, height) in detected_faces:
                cv2.rectangle(image, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 0, 0), 2)

            cv2.imshow("Face Detection Window", image)

            key = cv2.waitKey(30) & 0xff
            # Stop the program on the ESC key
            if key == 27:
                break

        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("Cannot open Camera")

if __name__ == "__main__":
    faceDetect()
```

## Installing MXNet

MXNet 컴파일에 최소 5시간 이상 소요되므로 nohup으로 백그라운드 작업을 걸어놓고 수행하는 것을 추천

2020년 7월 기준으로 1.6.0이 최신 버전이지만, 호환성 측면에서 좋지 않고\(단, GluonNLP를 사용한다면 1.6.0 권장\) SageMaker도 현재 1.4.1을 사용하므로 1.4.1 설치 권장 \(1.5.1도 무방하며, 최근 포럼글에서는 1.5.1을 더 권장함\). 참고로, MXNet은 하위호환성 및 상위호환성이 없기에 동일한 버전으로 맞춰야 함.

* 1.6.0 : Precompiled 라이브러리 기준으로는 CUDA 10.2이며, 컴퓨터 비전에서는 몇 가지 문제들 발생하기에 추천하지 않음. 단, 자연어처리 학습을 위해 GluonNLP 설치 시에는 1.6.0만 사용해야 함
* 1.4.1~1.5.1: 추천 버전 & CUDA 10.0
* 1.2.1: Outdated & CUDA 9.0

#### 1\) swap 생성 \(Jetson Nano에 기본 내장된 4기가 메모리로는 부족하므로\)

```bash
fallocate -l 8G swapfile
sudo chmod 600 swapfile
sudo mkswap swapfile
sudo swapon swapfile
```

#### 2\) 의존성 패키지 설치

```bash
sudo apt-get update
sudo apt-get install -y git build-essential libatlas-base-dev graphviz
sudo pip install --upgrade setuptools
sudo pip install numpy==1.15.2
sudo pip install graphviz jupyter
```

#### 3\) Build MXNet \(약 5-6시간 소요됨\)

`make -j3` 옵션으로 실행 시 간헐적으로 다운될 수 있으므로, 만약 중간에 다운되면 `make -j2`로 변경

```bash
git clone https://github.com/apache/incubator-mxnet.git --branch v1.4.x --recursive
cd incubator-mxnet/
```

```bash
cp make/config.mk .
sed -i 's/USE_CUDA = 0/USE_CUDA = 1/' config.mk
sed -i 's/USE_CUDA_PATH = NONE/USE_CUDA_PATH = \/usr\/local\/cuda/' config.mk
sed -i 's/USE_CUDNN = 0/USE_CUDNN = 1/' config.mk
sed -i '/USE_CUDNN/a CUDA_ARCH := -gencode arch=compute_53,code=sm_53' config.mk
```

```bash
make -j3
```

#### 4\) Install MXNet

```bash
cd python
sudo python setup.py install
cd ..
export MXNET_HOME=$(pwd)
echo "export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
```

#### 5\) `libopencv-dev` 설치

OpenCV를 인스톨해도 일부 의존성 파일들을 MXNet에서 찾지 못하므로 `libopencv-dev` 설치로 해결 가능. 참고로 libopencv-dev 버전은 4.1.1로 버전 충돌 문제 없음.

```bash
sudo apt-get install libopencv-dev
```

#### 6\) \(Optional\) 테스트

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2,3), mx.cpu())
>>> print(a*2)

[[2. 2. 2.]
 [2. 2. 2.]
<NDArray 2x3 @cpu(0)>

>>> import mxnet as mx
>>> a = mx.nd.ones((2,3), mx.gpu()) # 처음 실행 시 약 30초 정도 딜레이됨
>>> print(a*2)

[[2. 2. 2.]
 [2. 2. 2.]
<NDArray 2x3 @gpu(0)>
```


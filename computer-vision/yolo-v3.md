# YOLO v3

## Algorithm

### Bounding Box Prediction

* 기본적으로 YOLO v2와 동일
* 학습 시에는 sum of squared error 사용
* Objectness score 예측 시 logistic regression을 사용; 어느 한 bounding box가 다른 bounding box보다 더 많이 ground truth object에 오버랩된 경우 값이 1이 됨 → 즉,  ground truth마다 bounding box가 1개만 할당됨!

### Class Prediction

* Softmax 대신 binary cross-entropy loss 사용
* 이 방식이 좀 더 복잡한 domain 상에서 label들이 많이 겹치는 분류 문제에 도움이 됨.
  * 예: 사람 & 여자가 label 셋에 모두 존재하는 경우 mutually exclusive하지 않음

### Predictions across Scales

* 3개의 다른 스케일의 box 예측 \(Spatial Pyramid\)
  * 416x416 영상의 경우 bounding box의 개수는 \(52 x 52\) + \(26 x 26\) + 13 x 13\)\) x 3 = **10647** 개

![](../.gitbook/assets/untitled%20%2813%29.png)

* 최종 layer에서 bounding box, objectness, class prediction 예측
* 예: COCO dataset
  * N x N x \[3\*\(4+1+80\)\]
  * N: grid 개수, 3: anchor box 개수
  * 4 bounding box offsets, 1 objectness score, 80 class predictions
* k-means clustering을 좀 더 정교하게 수행
  * 9개의 cluster를 사용해서 3개의 scale에 대해 임의로 anchor box dimension 할당
  * COCO dataset의 경우 \(10×13\), \(16×30\), \(33×23\), \(30×61\), \(62×45\), \(59×119\), \(116×90\), \(156×198\), \(373×326\) 사용

### Feature Extractor

* Darknet-53 사용 \(resnet과 동일한 성능인데 속도는 2배 빠름,  주로 3x3, 1x1 Conv와 skip-connection으로 이루어져 있음\)

![](../.gitbook/assets/untitled-1%20%283%29.png)

* ResNet-101보다 1.5배 빠르고 성능은 더 좋고 RsesNet-152와 성능은 비슷하면서 2배 이상 빠름

![](../.gitbook/assets/untitled-2%20%287%29.png)

### Training

* Fulll image를 모두 사용하고 hard negative mining을 사용하지 않았음
* why? background 클래스에만 데이터가 너무 많은 class imbalance가 생겨서 hard negative mining을 사용하는데 YOLOv3은 별도의 background 클래스를 쓰지 않으므로

## References

* Paper
  * [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
* Blog
  * [https://taeu.github.io/paper/deeplearning-paper-ssd/](https://taeu.github.io/paper/deeplearning-paper-ssd/)
  * [https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)


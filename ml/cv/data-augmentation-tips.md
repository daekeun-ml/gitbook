# Data Augmentation Tips

## 1. Albumentations Library

### 특장점

* 다양한 Augmentation 기법 및 빠른 속도로  augmentation pipeline을 구성할 수 있는 오픈 소스 라이브러리
* Kaggle Master들이 작성한 라이브러리로 Kaggle, topcoder, CVPR, MICCAI의 많은 컴피이션에서 활발히 사용 중
* 다양한 Pixel 레벨 변환 및 Spatial 레벨 변환을 지원하며, 확률적으로 augmentation 여부를 선택할 수 있고OneOf block으로 선택적으로 augmentation 방법을 선택 가능

### Benchmarking

* Intel Xeon Platinum 8168 CPU에서 ImageNet 검증셋의 2,000개 이미지에 대한 벤치마킹 수행
* 아래 표는 단일 코어에서 처리하는 초당 이미지 수로 albumentation이 대부분의 transform에서 2배 이상 빠름

### Code Snippets

#### How to use

* PyTorch의 torchvision과 매우 유사 (5\~10분이면 익힐 수 있음)
* Documentation: [https://albumentations.readthedocs.io/en/latest/](https://albumentations.readthedocs.io/en/latest/)
* Colab에서 쉽게 테스트 가능: [https://colab.research.google.com/drive/1JuZ23u0C0gx93kV0oJ8Mq0B6CBYhPLXy#scrollTo=GwFN-In3iagp\&forceEdit=true\&offline=true\&sandboxMode=true](https://colab.research.google.com/drive/1JuZ23u0C0gx93kV0oJ8Mq0B6CBYhPLXy#scrollTo=GwFN-In3iagp\&forceEdit=true\&offline=true\&sandboxMode=true)

```python
torchvision_transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Same transform with torchvision_transform
albumentation_transform = albumentations.Compose([
    albumentations.Resize(256, 256), 
    albumentations.RandomCrop(224, 224),
    albumentations.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
    albumentations.pytorch.transforms.ToTensor()
])

img = img[:,:,np.newaxis]
img = np.repeat(img, 3, axis=2)
torchvision_img = torchvision_transform(img)
albumentation_img = albumentation_transform(image=img)['image']
```

#### Probability Calculation

```python
from albumentations import (
   RandomRotate90, IAAAdditiveGaussianNoise, GaussNoise, Compose, OneOf
)
import numpy as np

def aug(p1, p2, p3):
   return Compose([
       RandomRotate90(p=p2),
       OneOf([
           IAAAdditiveGaussianNoise(p=0.9),
           GaussNoise(p=0.6),
       ], p=p3)
   ], p=p1)

image = np.ones((300, 300, 3), dtype=np.uint8)
mask = np.ones((300, 300), dtype=np.uint8)
whatever_data = "my name"
augmentation = aug(p1=0.9, p2=0.7, p3=0.3)
data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
augmented = augmentation(**data)
image, mask, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]
```

* $$p_1$$: augmentation 적용 여부 판단 (1일 경우에는 항상 augmentation 적용)
* $$p_2$$: 90도 회전 여부 결정
* $$p_3$$: `OneOf` 블록 적용 여부 결정 (블록 내 모든 확률을 1로 정규화한 다음, 정규화한 확률에 따라 augmentation 선택)
  * 예: `IAAAdditiveGaussianNoise`의 확률이 0.9이고 `GaussNoise`의 확률이 0.6이면 정규화 후에는 각각 0.6과 0.4로 변경됨
    * $$0.6 = (0.9 / (0.9 + 0.6)),  0.4 = (0.9 / (0.9 + 0.6))$$&#x20;
* 각 augmentation은 아래와 같은 확률이 적용됨
  * `RandomRotate90`: $$p_1 * p_2$$&#x20;
  * `IAAAdditiveGaussianNoise`: $$p_1 * p_3 * 0.6$$&#x20;
  * `GaussianNoise`: $$p_1 * p_3 * 0.4$$&#x20;

## 2. CutMix

### Background

* Cutout: 훈련 이미지에서 일부 영역을 검은 색 픽셀 및 랜덤 노이즈 패치로 오버레이하여 제거 → 일부 경우에 잘 동작하지만  작은 object나 중요한 영역에 대한 정보 손실이 많이 발생하는 문제
* Mixup: 이미지와 정답 레이블의 선형 보간(linear interpolation)을 통해 두 개의 샘플을 혼합 → 지나친 smoothing 효과로 object detection에서 그리 좋지 않음
* CutMix: 두 이미지의 정보를 모두 살려 보자는 취지
* <img src="../../.gitbook/assets/Untitled 1 (4).png" alt="" data-size="original"> <img src="../../.gitbook/assets/Untitled 2 (9).png" alt="" data-size="original">&#x20;

### **Algorithm**

* 먼저, 훈련 이미지, 클래스, 각 클래스에 해당하는 훈련 샘플을 아래와 같이 정의하면,

$$
(x, y): \text{Training image, label} \\ (A, B): \text{Training class} \\ (x_A, y_A), (x_B, y_B): \text{Training sample}
$$

*   두 개의 샘플 이미지인 $$(x_A, y_A), (x_B, y_B)$$를  사용하여 새로운 샘플 $$(\tilde{x}, \tilde{y})$$를 생성할 수 있다.&#x20;

    ($$\odot$$: element-wise multiplication)

$$
\tilde{x} = \mathrm{M} \odot x_A + (1 - \mathrm{M}) \odot x_B \\ \tilde{y} = \lambda{y_A} + (1 - \lambda)y_B
$$

* $$M$$: 0이나 1로 표현되는 $$W*H$$ 차원의 binary mask 영역으로 어느 부분을 mix할 것인지 결정
  * $$M$$의 영역은 $$\lambda$$파라메터에 의해 결정되며, 두 이미지에서 잘라낼 영역을 알려 주는 bounding box 좌표 B를 가져와서 샘플링
  * $$x_A, x_B$$라는 이미지가 있을 때, $$x_A$$내의  특정 bounding box 영역을 $$x_B$$ 이미지로부터 가져와서 붙임
  * 즉, $$x_A$$의 bounding box 영역 B가 제거(crop)되고 그 영역은 $$x_B$$ 의 bounding box B에서 잘린 패치로 대체됨(paste).
* $$\lambda$$: mixing ratio로 베타 분포에 의해 결정
* 이를 수식으로 표현하면 아래와 같다.

$$
\mathrm{B}: \text{Bounding box coordinates } (r_x, r_y, r_w, r_h) \\ r_x \sim \text{Unif }(0, W), r_w = W\sqrt{1-\lambda}, \\ r_y \sim \text{Unif } (0, H), r_h = H\sqrt{1-\lambda}
$$

* $$r_x, r_y$$는 bounding box 중심 좌표이며, uniform distribution에 의해 결정됨
* $$r_w, r_h$$는 bounding box의 너비 및 높이로, 이 수식을 통해 cropped area ratio $$1-\lambda$$는 아래와 같이 계산할 수 있다.&#x20;
* $$\dfrac{r_w r_h}{WH} = 1 - \lambda$$&#x20;

### Code Snippets

#### Bounding box 좌표 생성

```python
def rand_bbox(size, lam):
    '''
    CutMix Helper function.
    Retrieved from https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    '''
    W = size[2]
    H = size[3]
    # 폭과 높이는 주어진 이미지의 폭과 높이의 beta distribution에서 뽑은 lambda로 얻는다
    cut_rat = np.sqrt(1. - lam)

    # patch size 의 w, h 는 original image 의 w,h 에 np.sqrt(1-lambda) 를 곱해준 값입니다.
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # patch의 중심점은 uniform하게 뽑힘
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
```

#### 실제 호출 예시 (Kaggle Bangali.ai Handwritten recognition)

```python
bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
# 픽셀 비율과 정확히 일치하도록 lambda 파라메터 조정  
lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

logits = model(inputs)
grapheme = logits[:,:168]
vowel = logits[:, 168:179]
cons = logits[:, 179:]

loss1 = loss_fn(grapheme, targets_gra) * lam + loss_fn(grapheme, shuffled_targets_gra) * (1. - lam)
```

## References

* Paper
  * Albumentations: [https://arxiv.org/pdf/1809.06839.pdf](https://arxiv.org/pdf/1809.06839.pdf)
  * CutMix: [https://arxiv.org/pdf/1905.04899.pdf](https://arxiv.org/pdf/1905.04899.pdf)
* Official
  * Albumentations: [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
  * CutMix: [https://github.com/clovaai/CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch)
* Blog
  * [https://towardsdatascience.com/cutmix-a-new-strategy-for-data-augmentation-bbc1c3d29aab](https://towardsdatascience.com/cutmix-a-new-strategy-for-data-augmentation-bbc1c3d29aab)
* Video Clip (강추)
  * Albumentations: [https://www.youtube.com/watch?v=n\_f6d4bPFME](https://www.youtube.com/watch?v=n_f6d4bPFME)
  * CutMix: [https://www.youtube.com/watch?v=Haj-SRL72LY](https://www.youtube.com/watch?v=Haj-SRL72LY)

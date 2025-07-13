---
{"dg-publish":true,"permalink":"/blog/ml/what-is-machine-learning/"}
---

기계 학습은 일반적인 프로그래밍 방식과 다르다.

일반적인 프로그래밍 방식은 데이터를 미리 코딩된 프로그램에 넣어 계산(computation)을 통해 output을 출력하지만 기계 학습은 Training(정답인 input과 output set을 학습시키는 과정)을 통해 만든 Program(aka "Model")에 새로운 데이터를 넣었을 때 적절한 output을 출력한다.


기계 학습은 크게 세가지로 분류된다.

- Supervised learning(지도 학습)

- Unsupervised learning(비지도 학습)

- Reinforcement learning(강화 학습)

## Supervised learning

우선 지도 학습은 데이터의 input과 output을 아는 상태에서 둘 사이의 관계를 유형적으로 학습한다.

지도 학습은 두가지로 분류된다. 

- Classification

- Regression

### Classification

분류는 각 데이터가 어떤 Class에 속하는 지 구분하는 작업으로 정답값이 categorical한 변수이다.

$$\hat{y} = \hat{f}(\mathbf{x})$$
`x`: input으로 임의의 다차원 데이터(예: 사진, 텍스트)
`y`: output으로 categorical한 데이터(예: 강아지, 고양이)

$\hat{f}$: 수학적으로 input x를 $\hat{y}$로 예측, 기계 학습을 가지고 학습시키고자 하는 모델(함수)

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$$

즉 분류는 위 input-output 데이터 셋을 Traning하여 $y = \hat{y}$ 가 되도록 $\hat{f}$를 찾아내는 과정이다.

### Regression

회귀는 각 데이터가 어떤 continuous variable에 가까운지 예측하는 작업이다.  
예측하기 때문에 정답값이 continuous한 변수라고 할 수 있다.
$$\hat{y} = \hat{f}(\mathbf{x})$$
`x`: input으로 임의의 다차원 데이터(예: 자동차의 다양한 정보)  
`y`: output으로 continuos한 데이터(예: 자동차의 출력, 가격)  
$\hat{f}$: 수학적으로 input x를 $\hat{y}$로 예측, 기계 학습을 가지고 학습시키고자 하는 모델  
$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$$
즉, 회귀는 위 데이터 셋을 Traning하여 $|y - \hat{y}|$ 가 최소가 되게 하는(즉, 0이 되도록) $\hat{f}$를 찾아내는 과정이다.
$\arg\min_{\hat{f}} (y - \hat{y})$: $|y - \hat{y}|$을 최소로 하는 $argumnet \;\;\hat{f}$

## Unsupervised learning

비지도 학습은 데이터의 output을 모르는 상태에서 input의 흥미로운 특징(insteresting structure)을 무형적으로 찾아내는 것이다.  

입력 데이터만 제공되고 그 데이터들을 기반으로 관계를 학습한다.

- Clustering

- Latent factors

## Reinforcement Learning

이후에...
---
{"dg-publish":true,"permalink":"/blog/world-model/world-model/"}
---

[1st World Model](https://arxiv.org/abs/1803.10122)

# Abstract

RL 환경의 generative neural network model을 building

논문에서의 World Model(WM)은 **비지도 학습** 방식으로 빠르게 train 가능.
- 환경의 압축된 공간적, 시간적 표현들을 학습하는 것

이 WM에서 뽑은 feature들을 agent의 입력값으로 사용함으로써 더 compact하고 simple한 정책(요구되는 task를 풀 수 있는 정책)을 학습할 수 있다.

agent는 WM에 의해 생성된 hallucinated dream 내부에서 완전하게 train하고 이 학습시킨 정책을 실제 환경으로 되돌려 놓을 수 있다.

# Introduction
## Mental Model이란?

인간들은 제한된 감각으로 인식할 수 있는 것을 기반으로 mental model이라는 것을 개발하였다.

Dynamic system의 대부는 mental model을 다음과 같이 정의함.

*"The image of the world around us, which we carry in our head, is just a model. Nobody in his head imagines all the world, government or country. He has only selected concepts, and relationships between them, and uses those to represent the real system."*

우리의 뇌는 일상 속에서 방대한 양의 정보를 처리하기 위해 이 정보들의 spatial & temporal들을 모두 추상적으로 표현한다.

뇌 속 예측 모델을 이해하는 방법은 일반적으로 "미래"를 예측하는 것이 아니라 현재 motor action이 주어진 미래 감각 정보를 예측하는 것이다.  

그럼 우리는 본능적으로 이 예측 모델처럼 행동할 수 있을 것이고 의식적으로 어떤 행동을 할 지 plan을 정할 필요 없이 직면한 위험에 대해서 빠르게 반사 행동(reflexive behaviour)을 수행할 것이다.

## 야구를 예시로 들어보자

![image-6.png](/img/user/image-6.png)

타자는 짧은 시간(시각 정보가 뇌에 도달하는 시간보다 짧음) 안에 베트를 어떻게 스윙할 것인지 정한다.
이런 상황에서 공을 빠르게 칠 수 있는 이유는 우리의 능력이 본능적으로 공이 언제(temporal) 어디로(spatial) 갈 지 예측할 수 있기 때문이다.

프로들은 무의식적으로 이 과정이 이루어짐
- 근육 내부 모델의 예측에 따라 적절한 temporal, spatial에서 베트를 반사적으로 스윙함

공이 오는 동안 계획(베트를 어떻게 위치시키고 언제 때릴 지 타이밍을 잡을 지)을 세우지 않고 무의식적으로 내부 Mental 모델이 자동으로 미래의 예측(공을 스윙하는 것)을 수행하고 그 예측에 따라 반사적으로 행동함

즉, 현재 input으로 들어오는 시각 정보가 아닌 "예측된 미래"에 근거해 행동함 

## 이제 RL, RNN 문제에 대입해보자

Large RNN은 데이터의 풍부한 spatial & temporal 표현을 학습할 수 있는 매우 expressive한 모델.
하지만 Model-free RL(MDP를 모르는 RL로 agent가 환경과 직접 상호작용하면서 학습하는 강화학습의 종류)은 적은 파라미터를 다루는 작은 neural network를 사용한다.

RL 알고리즘은 신용 할당 문제에 의해 병목 현상이 발생한다. 이 신용 할당 문제는 전통적인 RL 알고리즘에서 large model의 수 천만개의 weight 학습을 어렵게 만든다.
- Credit assignment problem
	- agent가 reward를 받았을 때 그 reward의 원인이 이전의 action이나 decision인지 식별하는 문제

그래서 실제로는 training 하는 동안 좋은 policy를 여러 번 반복하기 위해 small network가 사용된다.

## 그럼 Large RNN 기반 agent를 이상적으로 학습시켜보자

역전파 알고리즘이 large neural network를 효율적으로 학습하는데 사용된다.

RL task를 다루기 위해 agent를 1. large world model과 2. small controller model로 나누어  large neural network를 학습시켰다.

우리는 비지도 학습 방식으로 agent의 World Model을 배우기 위해 large neural network를 학습시킴
- 보상 없이 sequence를 보고 세계를 이해하도록 훈련
그리고 이 world model이 만든 representation(latent vector z, hidden state h)을 사용하여 task를 수행하는 것을 배우기 위해 smaller controller model을 학습하였다. => policy 학습

small controller는 적은 수의 파라미터를 학습하기 때문에 보상의 원인을 추적하는 신용 할당 문제를 쉽게 다룰 수 있다.
- 이때 larger world model을 통해 capacity와 표현력을 잃지 않는다.

agent는 world model를 통해 세상을 간접적으로 학습시킴으로써 그 task를 수행하는데 compact한 policy를 효율적으로 학습할 수 있다.


# Agent Model

사람의 인지(cognitive) 시스템으로부터 영감 받은 간단한 모델
이 모델에서 agent는 세가지 요소를 갖고 있다.
1. V: small representative code를 보는 것을 압축하는 visual 요소
2. M: 과거 정보를 바탕으로 미래에 대한 예측을 하는 memory 요소
3. C: visual 요소와 memory 요소로부터 만들어진 표현에 따라 취할 행동을 결정하는 decision-making 요소

![image-7.png](/img/user/image-7.png)

## 2.1 VAE(V) Model

환경은 agent에게 각 time step마다 high dimensional input observation을 준다.
- 보통 video sequence에서 하나의 2D 프레임이 input으로 들어감

V은 각 input 프레임으로부터 추상적이고 압축된 representation을 배운다.

논문에서는 V를 VAE(Variational Auto Encoder)로 사용하여 각 input image를 small latent vector z로 압축한다.
![image-8.png](/img/user/image-8.png)

## 2.2 MDN-RNN (M) Model

V의 역할이 각 프레임에서 agent가 본 정보를 압축하는 동안 우리는 다음에는 어떤 상황이 일어날 지 궁금하다.
이때 M이 미래를 예측하는 역할을 한다.

| 구성 요소                              | 설명                                                   |
| ---------------------------------- | ---------------------------------------------------- |
| **RNN (Recurrent Neural Network)** | 시간 순서에 따라 정보를 처리하며, 과거의 상태를 기억하고 현재 입력을 반영해 출력을 생성   |
| **MDN (Mixture Density Network)**  | 출력을 단일 값이 아닌 **확률 분포**(여기서는 Gaussian 혼합 분포)로 표현하는 모델 |
이는 시계열의 다음 값을 하나의 scalar 값을 예측하지 않고 그 값이 어디쯤 나올 확률이 높을 지 분포 형태로 예측

강화학습에서는 Return이 불안정함
- 같은 action을 취하더라도 다음 state가 확정적이지 않음
- 따라서 MDN은 확률 분포로 모델링함 


M이 V가 생산할 것으로 예측되는 미래의 z vector를 예측하는 역할을 한다.
많은 complex한 환경들이 자연에서는 확률론적이므로
결정론적인 z 대신 확률 밀도 함수(PDF)을 출력하도록 RNN을 훈련시킨다.

![image-9.png](/img/user/image-9.png)

p(z)를 가우시안 분포의 mixture로 근사하고 RNN이 현재와 과거의 정보가 주어지면 다음에 오는 vector $z_{t+1}$의 확률 분포를 출력하도록 훈련시킨다. 

RNN은 $P(z_{t+1} | a_t, z_t, h_t)$를 모델링한다.
$z_t$는 현재 이미지에서 관측된 latent vector(VAE의 인코딩 결과)
$a_t$는 time step t에서 취할 행동
$h_t$는 time step t에서의 RNN's hidden state

output
$P(z_{t+1})$: Gaussian Mixture Model 파라미터로 표현 
- 여러 개의 평균 $u_i$, 분산 $\sigma_i$, 혼합 비율 $\pi_i$

이는 학습 단계에서 출력되며 RNN이 단순히 '다음 상태는 이거야'라고 고정된 값을 출력하는 게 아니라,  
다음 상태가 어떤 '확률적 공간' 안에 있을지를 예측하도록 유도하기 위함

이렇게 하면, 현실에서의 다양성과 불확실성을 모델이 포용 가능  

즉 z는 하나의 벡터가 아닌 여러 가우시안 분포에서 하나를 샘플링함

위 과정의 샘플링을 하는 동안 모델의 불확실성을 제어하기 위해 `temperature parameter `$\tau$를 조정

|τ\tau 값|의미|효과|
|---|---|---|
|낮음 (e.g., 0.1)|모델이 더 확신 있게 예측|단일 모드로 수렴 → 예측이 "결정적"임|
|높음 (e.g., 1.2)|더 많은 불확실성 허용|다양한 결과 샘플 가능 → "dream world"가 더 랜덤해짐|
논문에서는 τ\tauτ를 조절해서 dream environment의 현실성/난이도 조절에 활용함

이 방식을은 MDN-RNN(Mixture Density Network)에서 사용하는 것들이다.
이 모델은 handwriting이나 sketches 같은 sequence generation problems에 적용된다.

![image-10.png](/img/user/image-10.png)

## Controller(C) Model

C는 환경에서 agent가 동작하는 동안 누적 보상(Return)을 최대화 하기 위해 취해야 할 course of actions를 결정

논문에서는 C를 가능한 작고 간단하게 만듬(credit assignment 문제를 피하기 위해)
그리고 C는 V, M과 분리하여 학습해 대부분의 agent's complexity를 world model에 학습하게 한다.

C는 각 time step마다  $z_t$와 $h_t$를 concat하여 선형 변환해 직접 $a_t$에 맵핑하는 간단한 선형 모델이다.

$$ \large a_t = W_c[z_t h_t] + b_c $$
$W_c$: weight matrix
$b_c$: bias vector

## V, M, C Together

![image-11.png](/img/user/image-11.png)

우리가 둔을 플레이 한다 생각해보자.
먼저 각 프레임(time step t)마다 이미지가 V로 input으로 들어가 $z_t$(compressed latent vector)가 생성될 것이다.

M의 input은 $z_t, a_t, h_t$ 이고 
output은 next hidden state $h_{t+1}$, $P(z_{t+1})$(학습 시)


C의 input은 $z_t$와 M에서 출력된 hidden state $h_t$를 concat
output은 action $a_t$
즉, 지금까지 관측된 state와 RNN이 기억하는 과거 정보를 바탕으로 가장 보상이 높을 것으로 기대되는 행동을 선택 

그리고 C는 action vector $a_t$를 출력하고 이는 환경에 영향을 준다.

그 후 M은 현재 $z_t$와 $a_t$를 input으로 M의 hidden state를 업데이트한다.(netx time step $h_{t+1}$을 만들기 위해)

수도코드
![image-14.png](/img/user/image-14.png)

world model은 시뮬레이션 환경으로 여기서 출력된 obs가 맨 처음 3개의 모델이 포함된 사진에서 스크린샷이다. 발표 시 이 코드와 해당 그림을 연관지어서 설명하자. 

# 3. Car racing experiment

이제 위에서 설계한 agent model을 가지고 car racing task에 적용해보자.

## 3.1 World Model for Feature Extraction

우리가 만드는 predictive world model은 유용한 spatial & temporal한 표현들을 추출하는데 도움을 준다.

### CarRacing-v0 environment
이는 자동차는 가만히 있고 배경이 아래로 내려오는 top down 방식이다.

Env: 
매 시도마다 자동차가 달리는 트랙은 랜덤하게 생성된다. 
agent는 제한 시간 내 가능한 멀리 갈 수록 보상을 받는다.

action space는 3개로
좌/우 핸들링, 엑셀, 브레이크가 있다.


V Model을 train하기 위해 환경에서 10,000개의 랜덤 rollout 수집

env를 여러번 탐험하기 위해 agent가 무작위로 action을 취하게 만든다.
해당 action과 그 action의 결과로 변한 환경의 obervation을 기록한다.

이 관찰된 각 frame의 latent space $z$를 학습하기 위해 위 데이터셋을 V에게 훈련시킨다.

VAE가 위 env에서 주어진 frame과 $z$의 디코더로부터 생성된 (재구성된) 버전 사이 차이를 최소화시킴으로써 각 frame을 낮은 차원의 latent vector z로 encoding하도록 train시킨다. 
- 이 재구성된 이미지가 원본과 얼마나 잘 맞는지를 기반으로 VAE가 학습됨


이렇게 훈련된 V는 time step t에서 각 frame을 $z_t$으로 전처리하여 M 모델을 학습시키는 데 사용할 수 있다.
이렇게 전처리된 데이터를 사용해 저장된 random action $a_t$와 함께 M 모델(MDN-RNN)은 가우시안 분포의 mixture처럼 $P(z_{t+1| | a_t, z_t, h_t})$를 모델링하는데 사용된다.



---
이 carRacing v0 실험에서 world model(V & M)은 환경으로부터 실제 보상 signal이 무엇인지 아에 모른다.

이 task는 단순히 관찰된 일련의 image frame을 압축하고 예측하는 것이다.

C 모델만 환경으로부터 reward 정보에 접근할 수 있다.

VAE를 사용해 각 time step $z_t$를 사용해 각 frame을 압축해 재구성하여 
agent가 rollout 동안 실제로 보는 정보의 품질을 시각화할 수 있다.
![image-13.png](/img/user/image-13.png)




## 3.2 Procedure

1. Collect 10,000 rollout from a random policy
2. Train V(VAE) to encode frame into $z \in R^{32}$
3. Train M(MDN-RNN) to model $P(z_{t+1} | a_t, z_t, h_t)$
4. Define Controller C as $a_t = W_c [z_t h_t] + b_c$
5. Use CMA-ES to solve for a $W_c$ and $b_c$ that maximize the expected cumulative reward
	- CMA-ES는 최적화 task에 적합한 알고리즘

| Model      | Parameter count |
| ---------- | --------------- |
| VAE        | 4,348,537       |
| MDN-RNN    | 422,368         |
| Controller | 867             |
## 3.3 Experiment Results

### Only V model

VAE만 사용해서 agent가 얼마나 잘 주행할 수 있는 지 평가하기 위한 비교 실험

즉 "temporal 정보를 담당하는 MDN-RNN 없이도 주행이 가능한가?"를 실험 한 것
MDN RNN을 사용하지 않으므로 $h_t$ 없이 C model은 $\huge a_t = W_c z_t + b_c$로  표현 가능

현재 car racing의 한 프레임의 압축 정보만 보고 action을 결정
-> 현재 프레임의 순간적인 시각 정보만을 기반으로 행동

| 실험 조건                         | 평균 점수 (100회) | 특징                       |
| ----------------------------- | ------------ | ------------------------ |
| V 모델만 사용 (C: 단순 선형)           | 632 ± 251    | 많이 흔들림(wobbly), 급코너에서 실패 |
| V 모델만 사용 (C: hidden layer 추가) | 788 ± 141    | 성능 향상, 그러나 여전히 완전하지 않음   |
V 모델만으로도 성능은 나쁘지 않지만 어느 정도 주행은 가능함

하지만 날카로운 코너에서 종종 실패하고 차가 부들부들 떨리는 wobbing 현상이 발생
- 이는 temporal 정보가 없어서 불안정한 것
	- 현재 순간의 이미지 표현만 담고 있기 때문


### V & M model 

이번 실험에는 VAE의 출력 $z_t$와 MDN-RNN의 hidden state $h_t$를 같이 사용

$\large a_t = W_c[z_t, h_t] + b_c$

$z_t$: 현재 spatial 정보 요약
$h_t$: 다음 상태 $z_{t+1}$를 예측하도록 훈련됨 -> temporal 정보를 사용해 미래에 대한 추론


앞선 wobbing 현상이 없었다.
곡선에서도 훨씬 안정적이며 민첩한 움직임이 가능ㅎ

논문에서는 이를 "인간이 직관적으로 코너를 예측해 운전하는 방식과 유사"하다고 설명
- 강화학습에서 사용되는 방식은 미래를 시뮬레이션(rollout)하고 
- 그 결과를 바탕으로 action을 선택하는 계획 기반 접근(Monte Carlo Tree Search)

- 하지만 $h_t$에 미래에 대한 확률적 정보가 내포되어 있어 별도의 계획 없이 직관적이고 빠른 반응 가능 

이 구조가 실제 인간처럼 동작한다.
우리가 빠르게 실제 차를 주행할 수 있는 이유는
- 도로의 방해물들을 하나하나 예측하지 않고
- 두뇌가 이미 축적된 경험을 바탕으로 즉각적 결정을 내리기 때문
- 이와 유사한 것이 $h_t$가 과거의 sequence를 내재화하여 빠르고 일관된 행동 결정이 가능


---
![image-15.png](/img/user/image-15.png)

CarRacing-v0 환경은 평균 900 이상이면 solved로 간주됨
World Models는 이 기준을 넘긴 최초의 사례

기존 RL 방식들은
- 입력을 정제하기 위해 다양한 **수작업 전처리**가 필요:
    - 윤곽선(edge) 추출
    - 여러 프레임을 쌓기 (frame stacking) → 시간 정보를 간접적으로 제공
        
- 이는 설계자의 도메인 지식과 복잡한 세팅에 의존한다는 단점

논문에서 WM의 작관적이고 단순한 접근:
- 이 논문에서는 raw 이미지 그대로 받아들여:
    - VAE는 공간 구조 인코딩
    - RNN은 시간 정보를 직접 학습
- 별도 전처리나 프레임 스태킹 없이도 효과적인 상태 표현 학습 가능


World Models는 단순한 구조 (V + M + C)로, 별도 전처리 없이도 **고난도 연속 제어 문제를 해결**
- Deep RL 방식이 고질적으로 겪는 문제들:
    - 학습 불안정성
    - 높은 샘플 수요
    - 과도한 전처리 필요  
        에 비해, 훨씬 **학습 안정적이며 해석 가능하고 확장 가능한 프레임워크**를 제공함

## 3.4 CarRacing Dream

agent가 실제 환경이 아닌 꿈(dream) 내 환경에서 주행을 하는 실험

실제 환경에서 주행하는 것이 아닌 agent가 학습한 World Model을 바탕으로 가상의 racing 환경을 생성하고 주행하는 실험


World Model의 구성 복습
- **V (VAE)**: 이미지 → 저차원 벡터 $z_t$
- **M (MDN-RNN)**: 시간 정보 기억 + 미래 예측 → $P(z_{t+1} | z_t, a_t, h_t)$
- **C (Controller)**: $z_t, h_t$→ 행동 $a_t$


CarRacing Dream의 핵심 흐름
1. 초기 상태로 시작: $z_0, h_0$
2. 에이전트(C)가 행동 $a_t$​ 선택
3. M이 다음 상태 분포 $P(z_{t+1})$예측
4. 이 분포에서 샘플링 → $z_{t+1}$
    → 이는 실제 이미지 관측 없이 "hallucinated observation" 역할
5. V의 decoder를 사용하면 실제 화면도 재구성 가능
6. 반복

---
그럼 앞선 실험과는 뭔 차인가???

3.3 맨 처음 실험은 World Model을 실제 환경에서 진행한 것

실제 openAI GYM의 carracing v0 환경에서 agent가 운전
world model은 obs를 압축하고 temporal 정보를 예측하는 역할
C가 $z_t$와 $h_t$를 바탕으로 행동을 결정
reward는 실제 환경에서 반환함

장점:
정답이 정해져 있으니 학습 결과의 신뢰성이 높고 모델에서 오류가 없음

단점:
시뮬레이션이라 비용이 큼
샘플 수요가 많아 비효율적

3.4 실험은 MDN RNN만으로 상상해 주행한 것이다.

M model이 스스로 미래 상태 $\large z_{t+1} \sim P(z_{t+1} | a_t, z_t, h_t)$를 생성

agent(C)는 이 dream을 현실처럼 받아드림

장점:
빠르고 저렴하게 시뮬레이션 가능
실제 환경 없이 학습 가능

단점:
M의 모델이 불안정하거나 편향되면 잘못된 학습이 이뤄질 수 있음
학습된 정책이 실제 환경에서는 잘 작동하지 않을 위험 존재


# VizDoom
## Learning inside of a Dream

에이전트는 진짜 Doom 게임이 아닌, 자기 모델이 만든 가짜 Doom에서 움직이며 학습함
특히 M이 직접 “죽음”을 예측할 수 있으므로 보상 구조도 완전히 내적으로 정의 가능

그럼 어떻게 agent가 실제 VizDoom을 한 번도 플레이 하지 않고 Dream 환경 속에서 학습할 수 있을까?


1.  데이터 수집 (랜덤 정책)
	- 실제 VizDoom: Take Cover 환경에서 **random policy**로 10,000 에피소드 수집
	- 총 200,000 프레임의 이미지(관찰)와 행동 로그 확보
	- 이 과정이 유일한 실제 환경 접근이며, 이후 학습은 전적으로 내부에서 진행됨
	- 목적: 고차원 이미지 입력을 낮은 차원 벡터로 변환해 모델 학습과 제어 효율성 향상
	- 목적: VAE와 RNN 학습에 사용할 지도 데이터 생성

2. VAE 학습
	- 각 프레임을 인코딩해 128차원의 $z_t$로 압축

3. MDN-RNN 학습
	- Input: $z_t, a_t$
	- output: next time step t+1의 latent vector $z_{t+1}$의 확률 분포 $P(z_{t+1})$ + Death flag
		- MDN을 사용해 출력 분포를 가우시안 믹스처로 근사
		- $P(z_{t+1} | z_t, a_t, h_t)$  
		- agent가 죽었는 지 예측하는 binary classifier branch도 함께 학습
		- 목적: 꿈 속에서 행동했을 때 다음 상태와 죽음 여부를 예측 가능 


4. Dream 환경 생성
	- M model은 $z_t$와 $a_t$를 받아 다음 상태 $z_{t+1}$을 샘플링하고 deat 여부 반환
	- 이를 반복하면 M만으로 만들어진 dream vizdoom sequence가 생성됨


5. Controller 학습
	- Input: $z_t, h_t$
	- output: $a_t$
	- CMA-ES를 사용해 reward를 최대화하는 파라미터 탐색


좀 더 정확하게 어떻게 V+M만으로 만들어진 Dream(world model)에서 Controller를 학습시킬 것인가?
- agent의 생존 전력(policy)를 어떻게 학습?


논문에서는 credit assignment 문제를 피하기 위해 controller를 간단하게 만듬
- linear or shallow MLP model

학습은 Gradient Descent가 아닌 CMA-ES(Covariacne Matrix Adaptation Evolution Strategy)
- 모델 파라미터 집합(즉, policy)을 population 단위로 반복적으로 샘플링 → 성능 평가 → 적응
- 각 샘플은 dream environment에서 일정 시간 동안 실행하여 생존 시간 측정
- 평균 생존 시간이 높은 파라미터 집합을 다음 세대에 반영


더 자세한 CMA-ES...
CMAES는 진화 알고리즘의 한 종류로 목적 함수의 기울기를 모를 때 많은 후보 솔루션을 생성하고 성능을 비교해가며 업데이트 하는 방식


이런 방식은 역전파 없이도 policy를 효율적으로 탐색할 수 있음


예시:
1. dream 환경은 M이 샘플링한 $z_{t+1}$을 순차적으로 생성
    
2. 에이전트는 이 dream 시퀀스를 따라가며 action을 선택함
    
3. death prediction은 M이 직접 수행 → episode 종료 판단
    
4. reward는 **살아있는 동안 step마다 +1** → 생존 시간 = episode reward

먼저 C Model의 파라미터 $\theta$의 초기 분포를 설정
- 평균 $μ$ = 0, 공분산 $\sum$ = 1

오래 살아남은(성능이 좋은) 정책 파라미터를 기반으로 $\mu$와 $\sum$을 업데이트
- 평균 방향으로 이동
- 공분산을 적응시켜 잘된 방향은 더 넓게, 안 된 방향은 수축


그럼 이제 이렇게 학습된 policy를 실제 환경에 전이해보자

우리가 dream 환경에서 학습한 정책을 실제 VizDoom 환경에서 적용하면 잘 작동할까?

즉 sim-to-real transformer를 다룸

Model based RL의 성공 여부를 판단해볼거임

1. dream environment 안에서 CMA-ES로 policy 학습 완료
2. 학습된 controller $C(z_t, h_t)$ --> $a_t$를 고정
3. 실제 VizDoom: Take Cover 환경에서 Rollout 수행
4. 실제 생존 시간을 측정해 transfer가 잘 되었는지 평가

Dreamer 환경에서 학습된 정책을 현실에서 100번 테스트해봄

통과 기준 750 step보다 훨씬 높은 평균 1100 step을 기록함
심지어 dream 속 환경보다 더 높은 점수를 얻음
- 이는 오히려 dream 환경이 더 어렵다는 뜻

왜 dream보다 더 높은 점수가 나왔을까?

이유:
V는 시각 정보를 압축하면서 몇 가지 디테일을 놓침
- 현재 몇 명의 몬스터가 등장했는 지 정확히 파악 X
- 그럼에도 불구하고 agent는 이 불안정한 세계에서 학습한 전략으로 현실을 탐색
이런 nightmare dream 환경에서도 잘 살아남는다..? 그럼 실제 환경에서는 훨씬 더 잘 작동함 

=> 세부적인 정확도가 부족해도 정책 학습에 꼭 필요한 정보만 잘 유지되면 충분함

고난도 훈련소에서 훈련받은 병사가 실제 전투에서 더 잘 싸우는 것과 비슷한 논리


## 4.5 Cheating the world Model

세계를 속이는 전략:
애들은 게임에서 버그나 허점을 이용해 쉽게 이기는 법을 찾곤 함
그러면 정작 게임의 핵심 스킬은 못 배움 

이는 agent가 dream에서 설계된 의도와 다른 방식으로 목표를 달성할 수 있음

실제 실험 중 몬스터가 불덩이를 아에 쏘지 못하게 하는 이상한 정책을 학습함
- 몬스터가 공격하려고 하면 agent가 이상한 위치나 타이밍으로 이동해 fireball이 생기기도 전에 사라지는 현상을 찾아서
- agent의 입장에서 보면 이는 마치 하나의 솔루션으로 받아드릴 수 있음

하지만 이는 dream world에서만의 버그이지 실제 VizDoom에서는 그런 버그가 일어나지 않음

Dream(M's World)는 현실을 완벽히 묘사하지 못함
- 그저 stochastic, approximate, learned model 일뿐...

예시:
- 실제 환경에서는 3명의 몬스터가 있어야 하는데,
- M이 생성한 dream에서는 1명일 수도 있고 위치가 틀릴 수도 있음
- 심지어 monster가 fireball을 쏘는 **시점**도 다를 수 있음

이로 인해 agent는 실제 환경이 아닌 Dream의 약점을 기준으로 행동을 최적화함
즉, Controller는 이 허점을 공격해 cheating policy를 학습할 위험이 있음

---
M 모델은 환경의 dynamics를 시뮬레이션하는 역할임
agent는 이 모델을 기반으로 dream env에서 학습하며 단순히 obs만 보는 것이 아닌 M model의 내부 hidden state $h_t$도 함께 사용함
- hth_tht​는 RNN 내부의 누적된 기억으로 과거 action 및 latent 상태를 요약한 정보임
- 즉, agent가 “환경의 내부 상태”까지 들여다보고 있는 셈


일반적인 게임이나 현실 환경에서는 agent가 오직 외부 관측(observation)만 얻음
하지만 여기서는 game engine의 내부 메모리(M의 hidden state)까지 들여다볼 수 있음.
마치 플레이어가 아니라 디버깅 모드에서 게임을 조작하는 해커처럼 동작하는 상황

- agent는 보상을 최대화하기 위해 외부 환경을 탐색하는 것이 아니라
- game engine 내부 상태(M의 hidden state)를 조작하는 방식으로 전략을 학습할 수 있음


- - M은 학습된 모델이므로 완벽하지 않고 현실의 법칙을 근사할 뿐
    - 따라서 agent는 이 모델이 잘못 작동하는 지점(약점)을 찾아 exploit하는 정책을 학습할 수 있음.
    - 
- “adversarial policy”란?
    - 현실에서는 성립하지 않지만, dream 안에서는 고보상을 얻을 수 있는 정책
    - 즉 모델을 속이는 정책

그럼 학습된 정책은 M이 정상적으로 작동하는 범위 내에서 다뤄져야함
만약 모델이 경험하지 않은 새로운 state(Out of Distributino)으로 가게 되면 엉터리 예측, 모델을 속이는 정책인 adversarial policy를 하게 됨

이 hidden state를 "본다" 라는 뜻은 
Dream 환경을 생성하는 M model 내부의 상태인 $h_t$를 agent의 policy로 입력받으니 직접 사용하는 것을 의미한다.


![image-16.png](/img/user/image-16.png)
이런 그림은 우리가 Dream의 환경이 어떤 지 Decoding 한 것이지 실제 학습할 때 agent는 $h_t$로 학습하여 RNN 내부 M Model의 vector 자체를 명시적으로 입력 받음

---
기존 Model based RL은 World Model처럼 환경 Dynamic를 학습한 논문은 많지만(첫 번째 CarRacing 실험)
M 모델을 Dream환경처럼 사용하지는 않음
- Dream이 완벽하지 않기 때문에 agent가 cheating할 수 있어서

초기 M model은 deterministic했음
- 같은 input(state, action)을 주면 항상 같은 next state를 출력
- 이는 현실의 불확실성을 담지 못함

그래서 Bayesian Regression을 사용해 uncertainty를 다룸
- 모델이 모르는 영역에서는 확신이 없다는 정보를 주고 agent가 그쪽으로 가는 걸 억제함

But,
- GP(Gaussian Process)는 계산 비용이 크고, scaling이 어렵고
- 확률 분포를 알더라도 agent가 여전히 확률이 낮은 예외 상황을 속일 수 있음
- 즉, uncertainty estimation은 완화책이지 완전한 해결책은 아님

그래서 Hybrid approach를 사용
1. 처음에는 Model Based(Dream 환경)에서 policy를 빠르게 초기화
2. 이후에는 Model Free(PPO, SAC)으로 실제 환경에서 Fine-Tuning

이유:
- dream 환경으로는 초반 성능 확보가 가능
- 하지만 모델이 불완전하므로 결국 실제 환경에서 조정 필요


---
그럼 얼마나 세심하게 Dream의 한계를 극복할 것인가?

C(agent)가 M을 속이지 않으려면 Deterministic 예측 대신 좀 더 Stochastic 분포로 예측이 필요함

Temperature $\tau$ 조절로 학습 환경의 불확실정을 인위적으로 주입해 일반화를 강화함

- τ↑ → 더 stochastic한 환경 → agent가 M의 허점을 exploit하기 어려움
- τ↓ → 더 결정론적인 환경 → 성능 향상 가능하지만 exploit 위험 ↑

근데 왜 Deterministic 환경이 M의 허점을 잘 exploit할 수 있을까?

- M이 완벽하지 않다면
    - 일부 $(z_t, a_t)$ 조합에서 잘못된 z_{t+1}을 예측할 수 있음

- 이게 deterministic이면
    - 그 잘못된 예측은 항상 같은 방식으로 반복됨
        

> → C는 학습 중에 “이런 조합이면 이상한 보상이 나오는구나”를 정확히 외워서 계속 그 방향만 반복하게 됨


---
잠시 생각해보면 VAE의 출력값인 latent vector z는 single diagonal Guassian Distribution을 따름
- $\mathcal{N}(\mu, \sigma^2 I)$
- 이걸 예측하기 위해 굳이 Mixture of Guassian을 써야되나?

실제 환경에서는 불연속적인 이벤트가 많음
- 둠에서 몹이 파이어볼을 쏠 지 안 쏠지 
- Mixture Density Model은 다양한 mode를 각각 다른 가우시안 분포로 모델링할 수 있음
- single Gaussian은 평균값에 집중해 여러 mode를 표현하지 못함

Single diagonal Gaussian은 프레임을 인코딩하기에 충분
RNN은 시간에 따라 변하는 state를 예측해야 함
여기서 Mixture of Gaussian은 다양한 시나리오의 전개를 분리된 mode로 자연스럽게 표현 가능


- τ=0.1이면 샘플링이 거의 평균에만 집중됨
    - mixture 중 한 개의 모드만 지속적으로 선택됨 → mode collapse 현상
        
- 이 경우 dream world에서는:
    - 몬스터가 총을 아예 안 쏘는 시나리오만 학습될 수 있음


다른 모드 (fireball을 쏘는 모드)가 mixture 안에 존재하지만
낮은 τ로 인해 그쪽으로 샘플링되지 않음
따라서 C는 “fireball 없는 세상”에 적응된 정책만 학습하게 됨


- dream world에서는 실수 없는 완벽한 정책이 나올 수 있음
- 하지만 실제 환경에서는:
    - fireball이 실제로 날아오고
    - C는 그 상황을 전혀 고려하지 못했기 때문에
    - 현실에서 완전히 무용지물이 됨 (심지어 랜덤 정책보다 못함)


- temperature τ는 MDN-RNN의 샘플링 확률 분포의 분산 조절자임
    - τ↑ → 더 stochastic한 예측
    - τ↓→ deterministic에 가까운 예측 (→ exploit 가능성 ↑)
        
- 따라서 다양한 τ값에 대해:
    - C가 얼마나 robust한 정책을 학습하는지
    - 그 정책이 실제 환경에서도 잘 작동하는지(transfer 가능성) 실험



- 실제 실험을 수행함:
    - τ=0.1,0.5,1.0,1.15,1.3 값을 설정하고,

    - 각각의 환경에서 학습된 controller CCC를 **현실 환경(actual environment)**에 적용
        
- 그 결과:
    - 너무 낮은 τ: deterministic 환경 → 현실에서는 잘 안 맞음 (overfitting)
    
    - 너무 높은 τ: noise가 많아 학습 자체가 어려움
        
    - 적당한 τ: 현실에서도 잘 작동하는 정책 생성

![image-17.png](/img/user/image-17.png)


적당한 stochasticity가 있어야 M model이 현실을 잘 근사하고 C가 현실에서 잘 작동하는 일반화된 정책을 학습


---
# 5. Iterative Training Procedure


World Model을 더 복잡한 환경에 확장하려면 어떡해야 될까?

지금까지는 한 번의 sequence 데이터 수집과 학습(iteration이라 하겠음)만으로 간단한 환경을 해결했지만 복잡한 환경은 지속적인 데이터 수집과 학습의 반복이 필요

간단한 환경이란?
- 랜덤 정책으로 수집한 데이터만으로 V, M, C 모델 모두 충분히 학습 가능

복잡한 환경이란?
- 특정 전략적 행동을 통해서만 관찰 가능한 상태가 존재
- 초기 world model이 이런 상태를 보지 못하면 잘못된 모델이 될 수 있음
- Exploration과 지속적인 observation 수집이 필요


## Iterative Training 절차 

1. 모델 초기화
	1. M과 C를 무작위 파라미터로 초기화
2. 실제 환경에서 Rollout
	1. 초기 C를 기반으로 실제 환경에서 시뮬레이션 수행
	2. 그동안 observation $x_t$, action $a_t$, done flag 저장
3. 학습
	1. M 학습:
		1. next state, action, reward, done flag를 확률적으로 예측하도록 학습시킴
		2. $P(x_{t+1}, r_{t+1}, a_{t+1}, d_{t+1} \mid x_t, a_t, h_t)$
	2. C 학습: 
		1. 학습된 M 내부에서 rollout 하며 보상을 최대화하는 policy 학습  
		2. 즉, Dream 환경에서 정책 학습
4. 반복
	1. 실제 환경에서 다시 평가했을 때 목표 성능이 안나오면
	2. 지금까지 학습된 C를 통해 새로운 데이터 수집을 다시 시작
		1. 이 과정을 통해 더 정확한 M과 똑똑한 C가 만들어짐



CarRacing, VizDoom같은 간단한 환경은 한 번만의 학습 루프로도 성능이 충분했음
하지만 더 어려운 탐험 기반 환경에서는 다음이 필요
- 전략적으로 행동하는 C
- M의 모델링 개선
- 장기적 계획이 가능한 policy


### 강화 방법: Intrincsic Motivation

Artificial Curiosity
- 만약 탐험을 통해 모델이 높은 정확도를 달성했는지 보상으로 사용

Information-seeking behaivor
- Agent가 모델이 아직 모르는 영역을 의도적으로 방문하도록 유도



어떻게 모델의 정확도를 측정?

M은 $P(z_{t+1} | z_t, a_t, h_t)$을 출력하는 확률적 시계열 모델
만약 모델이 해당 상황을 잘 학습하지 못하거나 본 적 없다. -> "예측이 부정확하다"


M의 예측 실패 -> 곧 낯선 환경

여기서 제한된 핵심 아이디어는 M의 예측 오류(loss)는 탐험 가치(curiosity reward)로 활용
즉 M이 잘 못 맞추는 상황일수록 agent에게 보상을 줌
"M이 몰라?" -> "뭐지 흥미롭네" -> "함 가봐"


- C가 M이 잘 모르는 지역을 탐험 → 새로운 z, a, r, done 데이터 수집
- 이 데이터를 다시 학습에 사용 → M의 표현력 향상
	- → 이것이 world model self-improvement loop



World Model M이 저차원적인 Moter Skill(걷기)을 흡수하면, Controller C는 그것을 기반으로 더 고차원적인 전략(달리기)을 학습할 수 있다

좀 더 자세히 설명하면 World Model에서 걷기같은 기본적인 Moter skill을 계속 반복하여 학습할수록 이런 패턴을 내면화하게 된다.

$(z_t, a_t, h_t) → z_{t+1}$

이 관계를 여러 번 보면서 "걷기"라는 스킬은 더 이상 외부 입력이 아닌 M 내부의 저장된 action sequence 중 하나가 되는 것


물론 이런 내면화가 되려면 M은 충분히 크고 복잡한 네트워크여야함

그리고 이제 C는 이런 기능을 분리할 수 있음

M: "걷기"라는 행동의 결과를 이미 학습
C: 더 이상 "걷기"를 조절할 필요가 없음
이제 "어디로 걷냐", "왜 걷냐" 같은 planning이 가능

예전에는 C가 ‘왼발 들어, 오른발 들어, 균형 맞춰’를 다 했지만
이제는 ‘앞으로 5m 가’라고만 하면 M이 걷는 행동을 자연스럽게 시뮬레이션 해주는 것과 같음.

요약:
M: Low-level motor skills (걷기, 점프, 움직임)의 dynamics 학습
C: High-level decision making (경로 계획, 목표 선택 등) 학습

![image-18.png](/img/user/image-18.png)


- **Rehearsal (반복)**  
    → 반복적인 활성화(복습) 과정을 통해 단기 기억을 장기 기억으로 강화
    
- **Consolidation (정착화)**  
    → 단기 기억이 뇌의 다른 구조로 옮겨져 안정적인 장기 기억으로 저장됨

최근 경험을 반복하는 것은 memory 정착화에 중요한 역할을 함


# Related Work
뭔소리고

# Discussion
agent가 시뮬레이션 latent space dream world 내부에서 task를 수행하도록 훈련시킬 수 있는 가능성을 보여줌
만약 계산량이 많은 게임 엔진을 실행하려면 게임 상태를 이미지 프레임으로 렌더링하거나 물리 법칙을 즉각적으로 계산하기 위해 무거운 컴퓨터 리소스를 사용해야됨

이렇게 agent를 훈련시키는 주기를 낭비하고 싶지 않고 시뮬레이션 환경에서 원하는 만큼 충분히 agent를 학습시키고 싶다.

물론 실제 world에서 학습하는 것은 매우 비용이 있으므로 world model에서 학습 후 real world에 정책을 적용한다.

like sim2real

WM에서 시각정보를 처리하는 V 모델은 VAE로 비지도 학습 방식으로 학습한다.
입력 프레임을 z로 인코딩하여 latent space로 저장한다.

하지만 비지도 학습이므로 무엇이 중요한 지 모른다.
즉 어떤 요소가 보상과 관련이 있는지 고려하지 않고 단순히 이미지를 잘 복원하는 방향으로만 학습한다.

- Doom 환경에서는 **중요하지 않은 벽돌 무늬**를 자세히 재현해냄
- 반면 Car Racing 환경에서는 **도로 타일같이 중요한 정보**를 잘 복원하지 못함


해결책은?
- VAE가 단독으로 이미지를 복원하는 것이 아니라
    
- **M 모델과 함께 학습**시키는 방법:
    - M은 미래 상태뿐 아니라 **보상도 예측**
    - 그러면 V는 보상 예측에 **도움이 되는 정보**에 집중하게 됨
        

장점:
- 이렇게 하면 task-relevant feature 를 중심으로 인코딩하게 유도할 수 있음
    
    
단점:
- V가 특정 task에 최적화되기 때문에
- 다른 task로 재사용이 어려움 → 새로운 task가 생기면 V를 다시 학습시켜야 함





- **world model M**은 **MDN-RNN**, 즉 LSTM 기반의 순환 신경망
    
- 이 모델은 **환경의 dynamics**를 학습하고 미래 관측값 $z_{t+1}$를 예측
    

한계:
- LSTM은 구조상 기억할 수 있는 정보의 양이 제한되어 있음.
    
- 반복적인 학습(iterative training)으로 **엄청난 양의 시퀀스 데이터**를 수집하더라도,
    
    - 그 모든 데이터를 내부 weight에 "압축하여" 기억하는 데에는 한계가 있음.
        
- 결과적으로, 학습 도중 중요한 경험을 잊어버릴 수 있음.

Catastrophic Forgetting
- 신경망이 과거에 학습한 내용을 잊고 새로운 데이터에만 집중하게 되는 현상
- 시퀀스 환경에서 과거 정보를 계속 유지하지 못하고 손실됨


그럼 더 큰 모델로 교체하자~ 
or 외부 메모리 장착


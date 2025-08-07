---
{"dg-publish":true,"permalink":"/blog/world-model/planet/"}
---

[planet](https://arxiv.org/abs/1811.04551)


강화학습에서의 전통적인 planning:
전체적인 환경 모델을 통해 여러 state의 예측을 수행하고, 이를 바탕으로 행동을 계획하는 방식 
즉, 환경 모델을 기반으로 미래 상태를 예측하고 다양한 시나리오를 고려하여 결정하는 것

반면, PlaNet에서의 planning: 
latent space에서 학습된 동역학 모델을 바탕으로 온라인으로 계획을 세우고 행동을 선택하는 방식
이는 이미지와 같은 고차원적인 데이터를 압축하고 그 데이터로부터 미래를 예측하여 계획을 세운다는 점에서 전통적인 강화학습과 차별화됨

| 항목                  | **World Models** (2018)                       | **PlaNet** (2019)                           |
| ------------------- | --------------------------------------------- | ------------------------------------------- |
| **전체 구조**           | VAE + MDN-RNN + Controller                    | CNN encoder + RSSM + CEM planner            |
| **환경 시뮬레이션 방식**     | Dream 환경 (dream rollout)에서 policy 훈련          | Latent space에서 online planning              |
| **학습 방법**           | model-free evolution (CMA-ES)                 | model-based online planning                 |
| **latent dynamics** | MDN-RNN (stochastic, no recurrence in latent) | RSSM (stochastic + deterministic recurrent) |
| **보상 학습**           | 단순 예측 (reward 모듈 없음)                          | reward predictor 학습 포함                      |
| **정책 학습**           | Controller (feed-forward NN) + CMA-ES         | Cross Entropy Method (CEM) 기반 online MPC    |
| **목표**              | Dream 환경에서 model-free policy 학습               | 실제 환경에서 model-based planning 수행             |
| **overshooting**    | 없음                                            | Latent Overshooting 도입                      |

| 항목          | World Models (MDN-RNN)            | PlaNet (RSSM)                                   |
| ----------- | --------------------------------- | ----------------------------------------------- |
| 기본 구성       | VAE → MDN-RNN (RNN + GMM)         | CNN encoder → RSSM                              |
| Stochastic성 | GMM (Mixture of Gaussians) 기반 샘플링 | Gaussian stochastic + deterministic (GRU)       |
| 순환성 처리      | RNN 하나에 다 포함                      | `ht`: deterministic recurrent, `st`: stochastic |
| 표현력         | 약간 모호한 분포 표현 가능                   | 구조적으로 구분 가능, 안정적 학습                             |
| 목적          | dream rollout을 생성                 | planning을 위한 latent dynamics 학습                 |

| 항목       | World Models                       | PlaNet                           |
| -------- | ---------------------------------- | -------------------------------- |
| 정책 학습 방법 | CMA-ES로 controller 학습              | 매 타임스텝마다 CEM으로 online planning   |
| 환경       | Dream 환경 안에서 학습                    | 실제 환경 관측 후 planning 수행           |
| 수행 방식    | dream rollout 기반 모델-free policy 학습 | model-based policy, 한 스텝마다 다시 계획 |

| 관점     | World Models                      | PlaNet                            |
| ------ | --------------------------------- | --------------------------------- |
| 철학     | "dream을 통해 가상의 환경에서 policy를 학습하자" | "latent 공간에서 모델 기반으로 즉시 행동을 계획하자" |
| 방식     | dream 환경에서 policy를 CMA-ES로 반복 진화  | 실제 환경에서 한 스텝마다 CEM으로 최적 행동 선택     |
| 모델의 역할 | simulation world로써의 역할            | 정확하고 효율적인 planning을 위한 도구         |









# Learning Latent Dynamics for Planning from Pixels

> Abstract

환경의 Dynamic이 알려진 control task에 대한 planning이 성공적이었다.

미지의 환경(현실)에서 planning을 활용하러면 agent는 world와의 상호 작용에서 Dynamic을 배워야 함

planning에 충분한 학습 dynamic 모델은 image-based domain을 정확하게 학습하기 어려움
- 예측 오차가 시계열로 누적되고 미래의 다양한 가능성을 반영하는 것도 힘들고 분포 외 상황(OOD)에서는 예측이 과도한 자신감을 가질 수 있는 문제...

Deep Planning Network(PlaNet):
- image에서 환경의 Dynamic을 학습하고 
- 위 모델을 기반으로 latent space의 fast online planning을 통해 action을 선택하는 model-based agent


latent space planning:
- 실제 이미지를 예측하거나 시뮬레이션하지 않고도 추상적인 latent state 상에서 미래 state와 feature를 예측하고 최적의 행동을 선택하는 방식


1. RSSM(Recurrent State-Space Model):
	- 기존의 latent sequence model에 deterministicc & stochastic transition을 동시에 통합한 구조 

2. Latent Overshooting:
	- 기존 VAE는 기반 학습 방식은 보통 1-step 예측 성능만을 최적화하기 때문에 장기적으로 보면 오차가 누적되어 성능이 떨어지는 문제가 있다.
	- LO는 여러 시점에 걸친 multi-step 예측을 latent space 상에서 직접 정규화하는 방식으로 장기 예측 정확도를 개선함


PlaNet은 pixel image input만을 사용하여
+ contact dynamic
+ partial observability
+ sparse reward
현실적으고 어려운 문제 조건이 있는 continuous control task를 해결

+다른 Model Free 방법들과 비교했을 때도 훨씬 적은 episode수로 이상의 성능을 달성

---
# Introduction

Planning은 Dynamic이 알려져 있는 의사 결정 문제(game, robot)에서 강력한 접근법임
But env(Dynamic)을 모르는 상태에서 Planning을 하려 하면
agent는 경험으로부터 dynamic을 배워야 함

"Dynamics를 알고 있다." => agent가 환경의 dynamic system 동작 방식을 이해하고 있다.
- 즉 agent가 환경 내에서 state이 어떻게 변하는지와 action이 state에 미치는 영향, 그리고 reward가 이 state 변화에 따라 어떻게 주어지는지에 대한 모델을 알고 있다.

Dynamic Model을 학습하는 것은 오래된 도전 과제임

핵심 문제점은 
1. model inaccuracy, 
2. accumulating errors of multi-step prediction, 
3. failure to capture muliple possible future, 
4. overconfident prediction outside of training distribution

---
학습된 모델(Model-based)을 사용한 Planning은 Model Free-RL보다 더 낫다
why?
1. Model based planning은 Model free에 비해 데이터 효율적임
	1. 더 풍부한 training signal(과거의 경험을 바탕으로 미래의 행동에 대한 예측) 활용
	2. Bellman backup을 통한 propagating reward가 필요 X
2. Action space를 찾기 위해 컴퓨팅 리소스를 늘려 성능 향상 
3. 학습된 Dynamic은 특정한 task에 의존하지 않기 때문에 같은 환경에서 다른 task로 잘 전달될 수 있음
	1. 로봇의 물리적 움직임은 task가 달라도 동일
	2. 따라서 Dynamic Model만 잘 학습해두면 task-specific한 보상에 따라 re-planning 수행하여 다양한 task에 빠르게 적용 가능
	3. Model Free 강화학습은 특정 작업에 특화된 방법을 학습하여 한 작업에서 학습한 내용이 다른 task에 잘 적용되지 않을 수 있음 


최근 연구들은 단순한 저차원 환경에서의 Dynamic을 배우는 걸 보여줌

그러나 보통 이런 접근법은 세계의 state와 reward function에 대한 제대로된 접근법을 사용할 수 없다고 가정함.

고차원 환경에서는 fast planning을 위해 compate latent space 내 Dynamic을 학습하기를 원함.

이런 latent model은 간단한 task(cartpole)에서만 성공함

---
Deep Planning Network은 model based agent model이다.
- Image(Pixel)에서 환경의 Dynamic을 학습하고 compact latent sapce에서 online planning을 통해 action을 선택하는 `model based agent`

PlaNet은 transition model을 사용해 환경의 Dynamics를 학습함  
- 이 transition model은 stochastic 요소와 deterministic 요소 모두 포함함
- 불확실성을 고려하면서 예측 가능한 변화를 효율적으로 모델링


또한 multi step prediction을 장려하는 새로운 generalized variational objective를 실험함
- Variational Objective는 잠재적인 변수를 추정하는데 사용되는 목적 함수로 PlaNet에서는 이를 multi-step prediction을 촉진하는 방식으로 새롭게 일반화하여 사용함



Planet은 학습된 모델로 계획한 이미 해결된 것들보다 더 어려운 image pixel로부터 continuous control task를 푼다.

## Contribution

![image-19.png](/img/user/image-19.png)
1. Planning in latent space:
	1. agent가 환경의 고차원적인 시각적 정보를 압축해 저차원의 latent vector에서 환경을 모델링하고 환경의 dynamics을 고려하여 효율적인 계획을 통해 행동을 결정함 
		- agent는 model free 알고리즘인 A3c나 D4PG보다 더 좋은 성능을 보임

2. Recurrent state space model:
	1. Deterministic & stochastic component를 사용해 latent dynamic model을 만듬
	2. 예를 들어, 자율주행차의 경우 **도로 상황**(예: 장애물이 있는지, 차선의 변화를 예측)이 **결정론적**일 수 있지만, **교차로에서의 예측 불가능한 상황**(예: 다른 차들의 예상치 못한 행동)은 **확률적**으로 모델링해야 함. 두 가지 요소가 결합되어야 **복잡한 환경에서 더 정확한 예측** 가능.

3. Latent overshooting:
	1. Multi step prediction을 포함하도록 standard variational bound를 일반화함
	2. variational bound는 모델이 데이터를 잘 예측할 수 있도록 학습을 도와주는 목적 함수로 먼 미래의 예측을 보다 정확하게 만들 수 있음.

---
# 2. Latent Space Planning

planning을 통해 unknown환경을 풀기 위해 환경으로부터 env dynamic을 직접 학습해야 함.

Planet의 전체 흐름은 단방향이 아니라 반복적인 루프 구조로
처음에는 간단한 랜덤 정책으로 시작
1. 현재까지 학습된 모델로 planning을 수행해 행동을 선택하고 env에서 데이터를 수집
2. 수집된 데이터로 latent dynamic model을 학습

이처럼 planning -> interaction -> model update 루프를 통해 agent는 점점 정확한 dynamic model을 얻고 이에 따라 planning의 질도 향상됨

즉, 탐색과 학습이 함께 이루어지는 온라인 모델 기반 학습 구조임

## Problem setup

실제 환경에서는 한 장의 이미지가 전체 상태를 나타내지 않기 때문에 부분적으로 관잘할 수 있는
Partially obervable Markov Decision Process(POMDP)로 모델링



discrete time step: $t$
hidden state: $s_t$
observated image: $o_t$
continuous action vector: $a_t$
scalar reward: $r_t$

 stochastic dynamics를 따름


Transition function: 
$$s_t \sim p(s_t | s_{t-1}, a_{t-1})$$

Observation function:
$$ o_t \sim p(o_t | s_t) $$

Reward function:
$$ r_t \sim p(r_t | s_t) $$

Policy:

$$ a_t \sim p(a_t | o_{\le t}, a_{<t}) $$


특별한 일반성의 손실 없이, 항상 동일한 초기 상태 $s_0$에서 시작한다고 가정

목표는 정책 policy를 학습하는 것
어떤 정책? -> 총 보상 $\large E_p [\sum_{t=1}^T r_t]$ 의 기대값을 최대화하는 정책

---
## Model based planning

PlaNet은 이전에 경험한 에피소드로부터 다음 3가지 모델을 학습함
Transition model: $p(s_t | s_{t-1}, a_{t-1})$
obervation model: $p(o_t | s_t)$
reward model: $p(r_t | s_t)$
을 학습함

정자로 쓰인 모델은 true dynamic model.
기울어진 italic 모델은 learned model(예측된 모델).

observation model은 training 시 loss 함수의 계산에 도움을 주는 풍부한 training signal을 제공함.
- 실제 관측 $o_t$와 예측된 $\hat{o}_t$ 사이 loss를 줄이는 방식으로 모델이 잘 학습되는 지 확인 
- But planning에 사용되지는 않음
	- why?": planning은 latent space에서만 수행됨


그리고 Encoder 또한 학습함
- agent는 observation 값만 직접 확인할 수 있기 때문에 현재 hidden state $s_t$를 직접 알 수 없음
- 이를 해결하기 위해 filtering 기법을 사용한 encoder $q(s_t | o_{\le t}, a_{<t})$를 학습함

이 encoder는 과거 observation 값들과 과거 action 값들을 바탕으로 현재 시점의 hidden state에 대한 확률 분포: bleife를 추론함
- 현재 환경이 어떤 상태일지에 대한 확률을 추측하는 역할



PlaNet은 일반적인 RL처럼 policy를 학습하지 않음
대신 미래의 가능한 행동 sequence를 여러 개 샘플링하거나 탐색한 후 그 중 가장 좋은 결과를 낼 것 같은 행동 sequence를 선택하는 방식으로 계획 기반 행동 결정을 함
- 이런 계획을 짜는 행위가 PlaNet에서의 policy라고 할 수 있음

이 Plannign 알고리즘은 Model Predictive Control임
- 미래 몇 단계까지의 행동은 계획하되 실제로는 첫 번째 행동만 실행하고 다음 시점에 다시 관측된 정보로 새롭게 계획을 세우는 방식
	- 매 시점마다 계획을 다시 세움으로써 환경 변화나 예측 오류에 유연하게 적응 가능


Model Free RL에서는 action을 결정하기 위해 value network나 policy network를 학습하지만
여기서는 이런 network 아에 안쓰고 학습된 world model만으로 계획을 세워 행동을 결정함


---
## Experience collection

초기 agent는 환경의 모든 state space를 탐험하지 못함

agent가 환경을 탐험하면서 점차적으로 새로운 경험을 수집하고 이를 바탕으로 dynamic model을 정교하게 refine해야함

- model training -> model based planning -> collect new data -> model retraining
- 루프 반복

알고리즘 1을 통해 완전히 학습되지 않은 초기 world mode을 사용해도 planning을 수행하고 이를 기반으로 환경에서 행동을 선택해 데이터를 수집한다

![image-28.png](/img/user/image-28.png)

1. 초기 random action으로 수집한 S개의 에피소드만 가지고 학습 시작
2. 모델 파라미터 $\theta$ 랜덤하게 초기화
3. 학습을 일정 횟수 C만큼 업데이트를 할때마다 하나의 new 에피소드를 model based planning을 통해 수집함
	- B개의 시퀀스 청크를 데이터셋 `D`에서 무작위로 샘플링:
    $$\{ (o_t, a_t, r_t)_{t=k}^{k+L} \}_{j=1}^{B}$$
- 논문 Equation 3에 정의된 손실 함수 $\mathcal{L}(\theta)$ 계산 (잠재공간에서 ELBO 기반 손실)
- 경사 하강법으로 파라미터 업데이트
	- 조금씩 new 데이터를 수집하며 모델을 점진적으로 개선하는 방식


4. new 에피소드를 수집할 때 action에 small Gaussian exploration noise를 추가함
	- RL에서 exploration vs exploitation을 다루는 것처럼 탐험을 위한 noise 첨가

한번 action을 선택하면 그 행동을 R번 반복함

이는 planning의 길이: Horizon을 줄이기 위함
- Planning이 길어지면 미래 예측의 불확실성이 증가하므로 반복을 통해 예측 부담을 줄이고 모델의 학습을 더 안정적으로 만들 수 있음



## Planning algorithm

Planning은 단순 미래 예측이 아닌
latent space에서 가능한 action들을 상상하고
가장 큰 누적 보상을 만들어낼 수 있는 action sequence를 고르는 최적화 문제 

![image-29.png](/img/user/image-29.png)

Bste action sequence를 찾기 위해 CEM(Cross entropy method) 사용

robustness 하여 이 알고리즘을 사용
+
Planning을 위한 True dynamic가 주어졌을 때 이 알고리즘을 사용하면 all considered task를 풀 수 있음

CEM은 확률 분포를 샘플링하고 상위 성능을 가진 샘플들로 다시 분포를 업데이트하는 진화적 최적화 방법.
- 여러 개의 샘플을 기반으로 하여 가장 높은 누적 보상을 줄 수 있는 action sequence의 분포를 추정(infer)하는 것




$$a_{t:t+H} \sim \mathcal{N}(\mu_{t:t+H}, \sigma^2_{t:t+H} I)$$

시간에 따라 달라지는 가우시안 분포 $N(\mu, \sigma^2)$를 action sequence $a_{t:t+H}$에 대해 정의


- $\mu_{t:t+H}$: 각 시간 단계의 평균 행동
    
- $\sigma_{t:t+H}^2$​: 분산, 단일 변수에 대해 독립적이므로 대각행렬


초기에는 $\mu = 0\sigma = 1$, 즉 **무작위 분포**에서 시작

이후 J개의 후보 행동 시퀀스를 이 분포에서 샘플링

각 행동 시퀀스를 모델로 평가 (rollout)
    

모델은 이 행동 시퀀스를 따라가며 잠재 상태 전이 + 보상 합산을 계산  
→ 누적 보상이 높은 순서로 평가할 수 있음

샘플된 J개의 시퀀스를 평가하고,  
그 중에서 가장 좋은 K개를 골라  
→ 그들의 평균과 표준편차로 다시 가우시안 분포를 업데이트

이런 과정을 I번 반복 후 planner는 현재 시점 t에서의 최적행동으로 $\mu_t$를 반환 

loacl optima에 빠지지 않기 위해 매 time step마다 다시 mean:0 / var: 1로 초기화


---
이제 학습된 모델로부터 action sequence를 평가하기 위해 다음 과정을 거침 

- 현재 시점의 belief (확률분포 형태의 잠재 상태)로부터 시작하여
- 그 행동 시퀀스를 따라 잠재 상태들을 rollout (상상)
- 각 상태에서 예측되는 보상의 평균값(mean reward)을 합산하여 평가

CE처럼 여러 action sequence를 동시에 평가하는 population based optimizer를 사용하기 때문에
한 action sequence마다 단 하나의 rollout만 고려해도 충분하다는 것을 발견함
=> 더 많은 다른 action sequence를 평가하는데 계산 자원을 집중할 수 있다. 

PlaNet에서는 보상 함수도 잠재 상태 $s_t$를 입력으로 하는 함수로 모델링했기 때문에
- 플래너는 이미지나 실제 관측값을 생성할 필요 없이 순전히 latent space에서만 동작 가능
=> 많은 수의 후보 action sequence를 빠르게 평가

> 이미지 디코딩을 하지 않고, 오직 잠재 공간의 상태 전이와 보상 예측만으로 플래닝을 수행하므로 매우 계산 효율적이라는 것. 이것이 PlaNet이 고속 플래닝을 가능하게 만드는 핵심 요소 중 하나


# 3. Recurrent state space model(RSSM)

플래닝을 수행하기 위해서 에이전트의 매 시간마다 수천 개의 행동 시퀀스를 평가해야 한다.

따라서 순수하게  latent space 내부에서만 forward 예측을 할 수 있는 RSSM을 사용함

비선형 kalman filter or sequential VAE를 사용하는가 싶지만
이런 기존 구조들 대신 미래 dynamic model 설계에 도움이 될 두 가지 발견을 강조함

- "transition model에서 stochatsic path와 deterministic path 둘 다 함께 존재하는 것이 성공적인 Planning에 매우 중요하다"


## Latent dynamic

PlaNet에서 다음 sequence를 다룸
$$ \{{o_t, a_t, r_t\}}^T_{t=1} $$
전형적인 typical latent state space
![image-30.png](/img/user/image-30.png)

hidden state ${s_t}_{t=1}^{T}$를 기반으로 image 와 reward가 어떻게 생성되는지(generative process) 정의 
- 환경이 실제로 존재하는 것처럼 이미지와 보상을 생성해내는 내부 모사 모델을 구성하는 것
- 이 모델이 학습되면, 미래를 상상하거나 플래닝을 수행할 수 있게 됨

Transition model
$$ s_t \sim p(s_t | s_{t-1}, a_{t-1}) $$

가우시안 분포를 따르며 평균과 분산은 feed forward neural network으로 학습됨


Observation model
$$ o_t \sim p(o_t | s_t) $$
가우시안 분포를 따르며 평균은 Deconvolutional neural network가 출력하고
분산은 항등 행렬(identity covariance)로 고정?됨



Reward model
$$ r_t \sim p(r_t | s_t) $$
scalar값 가우시안 분포이고 평균은 MLP로 학습됨
분산은 1로 고정 

가우시안 분포의 log likelihood는 MSE와 거의 동일
- 학습 시 log-likelihood 최대화 = MSE 최소화


## Variational encoder

PlaNet은 비선형 구조이므로 hidden state의 posterior를 정확하게 계산하는 것이 불가능함


대신 아래처럼 근사된 posterior를 추론함

$$q(s_{1:T} \mid o_{1:T}, a_{1:T}) = \prod_{t=1}^{T} q(s_t \mid s_{t-1}, a_{t-1}, o_t)
$$

전체 state sequence $s_{1:T}$에 대한 근사 posterior 분포를
시간별 조건부 분포 $q(s_t | s_{t_1}, a_{t-1}, o_t)$들의 곱으로 분해
- 즉 순차적으로 상태를 업데이트해가며 추론하는 구조

각 time step의 posterior 분포는 Diagonal Gaussian 분포(변수 간 독립)
평균과 분산은 
- $o_t$을 입력으로 하는 CNN
- $s_{t-1}, a_{t-1}$: 과거 정보 -> MLP 처리
- 이들을 결합해 현재 hidden state s_t에 대한 분포의 파라미터 출력

학습 시 filtering posterior(지금까지의 과거 observation 값과 action만을 사용해 현재 state를 추론하는 posterior 분포)를 사용 => planning에 사용되는 구조와 맞추기 위함

현실에서의 planning은 미래 관측값을 알 수 없음(과거 정보만 사용해야 함)


## Training objective

VAE 학습
- log-likelihood를 최대화 하는 것
- 직접 계산하기 어렵기 때문에 Jensen's Inequality를 이용해 ELBO(evidence lower bound)를 대신 최적화함


$$\ln p(o_{1:T} \mid a_{1:T}) \geq \sum_{t=1}^T \mathbb{E}_{q(s_t \mid o_{\leq t}, a_{<t})} \left[ \ln p(o_t \mid s_t) \right] - \mathbb{E}_{q(s_{t-1} \mid o_{<t}, a_{<t})} \left[ \mathrm{KL}(q(s_t \mid o_{\leq t}, a_{<t}) \| p(s_t \mid s_{t-1}, a_{t-1})) \right]$$

좌변: log likelihood:
- 데이터(이미지 시퀀스)의 **진짜 확률을 최대화**하고 싶은 대상
    
- 하지만 계산이 어려워서 우변의 ELBO를 대신 최대화


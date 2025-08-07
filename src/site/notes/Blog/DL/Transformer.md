---
{"dg-publish":true,"permalink":"/blog/dl/transformer/"}
---

김성범 교수님의 유튜브 강의를 공부하고 남긴 정리글.
https://www.youtube.com/watch?v=a_-YgMO0u0E
# Introduction

Encoder와 Decoder 사이 관계에 중점을 주어야하는 부분들이 존재한다.
이 부분들에 가중치(weight)를 주는 것을 Attention이라 부름(Between)

`Encoder 내 weight`, `Decoder 내 weight`를 주는 것을 Self-attention이라 함(Within)

전통적인 Attention 방식은
"I love you" => "나는 너를 사랑해"
위 문장에서 "I"와 "나는" 이라는 관계를 Attention이라 한다면

Self-Attention 방식은
"나는 학생이다." 라는 문장 내 "나는"과 "학생"은 어떤 관계가 있는 지를 Self-Attention이라 한다.


Transformer는 어려 개의 Encoder와 Decoder로 구성되어 있으며 Input data가 Encdoer 여러 개를 거치고 마지막 Encdoer에서 Decoder로 정보를 전달한다. 
![image.png](/img/user/image.png)

Encoder:
Multi-Head Attention에서 "Attention"은 Encoder 안에서 sefl-attention을 의미한다.
![image-1.png](/img/user/image-1.png)


Decoder:
Masked Attention은 Decoder의 Self-Attention이고 그 위 두번째 Attention은 Tradition Attention임
![image-2.png](/img/user/image-2.png)


정리하면 아래와 같다.
![image-3.png](/img/user/image-3.png)


**Embedding:** 단어(Token) 형태의 데이터를 수치로 변환
- 초기에는 one-hot vector 형태로 입력되며 embedding layer를 통해 학습
- 유사 단어는 유사한 값을 지니도록 embedding 수행



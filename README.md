# About Batch Normalization
Tutorial of Batch Normalization

- BatchNorm 이 잘 되는 이유를 2015년에는 data-shift 와 관련지어서 설명한다. 즉, BN 레이어가 data-shift 를 막아주기 때문에 학습이 잘 된다는 것.
- 하지만, 2018년 MIT에서는 data-shift 가 아닌 BN 레이어가 loss function을 smoothing 하기 때문에 학습이 더 잘 된다고 해석한다. 
- 최근에는 BN이 잘 되는 이유를 loss function smoothing 으로 보고 연구가 진행되는 것 같다. 
- 예시로 BN-layer 없이 유사한 효과를 내기위한 논문에서는([High-Performance Large-Scale Image Recognition Without Normalization, DeepMind](https://arxiv.org/abs/2102.06171)) BN의 효과를 살리기 위해 gradient clipping을 사용해서 exploding gradient을 막는다. 




***
### Reference 
[1] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 2015, PMLR](http://proceedings.mlr.press/v37/ioffe15.html) / 초기 BN 논문은 Internal Covariate Shift(ICS)를 줄여서 성능을 높이기 위해 고안 됨.  <br/>
[2] [How Does Batch Normalization Help Optimization?, 2018, NeurIPS](https://papers.nips.cc/paper/2018/hash/905056c1ac1dad141560467e0a99e1cf-Abstract.html) / BN이 왜 잘 되는지 원인을 분석한 논문. <br/>
[3] [Batch Normalization - EXPLAINED!, CodeEmporium](https://youtu.be/DtEq44FTPM4) / <br/>
[4] [High-Performance Large-Scale Image Recognition Without Normalization, 2021, arXiv](https://arxiv.org/abs/2102.06171) / BN 레이어 없이 그러한 효과를 내는 방법 제안<br/>

---
layout: default
---
<!-- Text can be **bold**, _italic_, or ~~strikethrough~~. -->

<!-- [Link to another page](./another-page.html). -->

## Overview
A list of research papers on out-of-distribution (OOD) generalization in time series. Existing studies categorize the problem from three key perspectives: data distribution, representation learning, and OOD evaluation. For more details, please refer to our survey paper, *"Out-of-Distribution Generalization in Time Series: A Survey."*

## Data Distribution
Real-world data distributions are often dynamic rather than static and frequently subject to various distribution shifts that challenge the assumptions made during training. Two common distribution shifts are covariate and concept shifts.

### Covariate Shift

*   [CVPR 2021] Out-of-distribution Detection and Generation using Soft Brownian Offset Sampling and Autoencoders [paper](https://openaccess.thecvf.com/content/CVPR2021W/SAIAD/papers/Moller_Out-of-Distribution_Detection_and_Generation_Using_Soft_Brownian_Offset_Sampling_and_CVPRW_2021_paper.pdf)
*   [ICLR 2022] Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift [paper](https://openreview.net/pdf?id=cGDAkQo1C0p)
*   [SMC 2022] Feature Importance Identification for Time Series Classifiers [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9945205)
*   [FUZZ 2023] An Initial Step Towards Stable Explanations for Multivariate Time Series Classifiers with LIME [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10309814)
*   [Inf. Sci. 2023] Explaining time series classifiers through meaningful perturbation and optimisation [paper](https://www.sciencedirect.com/science/article/pii/S0020025523009192)
*   [Neural Networks 2024] SEGAL time series classification - Stable explanations using a generative model and an adaptive weighting method for LIME [paper](https://www.sciencedirect.com/science/article/pii/S0893608024002697/pdfft?md5=3f81e6d7a6bddcb6857d94aa6ab04937&pid=1-s2.0-S0893608024002697-main.pdf)
*   [VR 2024] Generating Virtual Reality Stroke Gesture Data from Out-of-Distribution Desktop Stroke Gesture Data [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10494175)
*   [AAAI 2024] Generalizing across temporal domains with koopman operators [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29604/31020)
*   [IJCAI 2024] Temporal Domain Generalization via Learning Instance-level Evolving Patterns [paper](https://www.ijcai.org/proceedings/2024/0470.pdf)
*   [ICML 2024] Connect later: Improving fine-tuning for robustness with targeted augmentations [paper](https://openreview.net/pdf?id=Uz4Qr40Y3C)
*   [ICML 2024] TimeX++: Learning Time-Series Explanations with Information Bottleneck [paper](https://openreview.net/pdf?id=t6dBpwkbea)

### Concept Shift

*   [AAMAS 2024] Rethinking out-of-distribution detection for reinforcement learning: Advancing methods for evaluation and detection, [paper](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1445.pdf)
*   [ICLR 2024] Disentangling Time Series Representations via Contrastive Independence-of-Support on l-Variational Inference [paper](https://openreview.net/pdf?id=iI7hZSczxE)

## Representation learning


### Decoupling-based Methods
Multi-Structured Analysis
*   [KDD 2024] Orthogonality Matters: Invariant Time Series Representation for Out-of-distribution Classification [paper](https://dl.acm.org/doi/pdf/10.1145/3637528.3671768)
*   [TKDE 2024] Disentangling Structured Components: Towards Adaptive, Interpretable and Scalable Time Series Forecasting [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10457027)
*   [NIPS 2020] Feature Shift Detection: Localizing Which Features Have Shifted via Conditional Distribution Tests [paper](https://proceedings.neurips.cc/paper/2020/file/e2d52448d36918c575fa79d88647ba66-Paper.pdf)
*   [IJCNN 2021] Unsupervised Energy-based Out-of-distribution Detection using Stiefel-Restricted Kernel Machine[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9533706)
*   [ACM TCPS 2022] Efficient Out-of-Distribution Detection Using Latent Space of β-VAE for Cyber-Physical Systems [paper](https://dl.acm.org/doi/pdf/10.1145/3491243)
*   [ACM TIST 2023] Out-of-distribution Detection in Time-series Domain: A Novel Seasonal Ratio Scoring Approach [paper](https://dl.acm.org/doi/pdf/10.1145/3630633)
*   [AAAI 2024] MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting [paper](https://ojs.aaai.org/index.php/AAAI/article/view/28991/29883)

Causality-Inspired
*   [CVPR 2021] Causal hidden markov model for time series disease forecasting [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Causal_Hidden_Markov_Model_for_Time_Series_Disease_Forecasting_CVPR_2021_paper.pdf)
*   [ICRA 2022] Causal-based Time Series Domain Generalization for Vehicle Intention Prediction [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9812188)
*   [ICML 2023] Neural Stochastic Differential Games for Time-series Analysis [paper](https://proceedings.mlr.press/v202/park23j/park23j.pdf)
*   [Sci. Robot. 2023] Robust flight navigation out of distribution with liquid
neural networks [paper](https://cap.csail.mit.edu/sites/default/files/research-pdfs/Robust%20flight%20navigation%20out%20of%20distribution%20with%20liquid%20neural%20networks.pdf)
*   [Inf. Sci. 2024] A causal representation learning based model for time series prediction under external interference [paper](https://www.sciencedirect.com/science/article/abs/pii/S002002552400183X)


### Invariant-based Methods
Invariant Risk Minimization:

*   [XXX 2024] XXX [paper](XXX)
*   [XXX 2024] XXX [paper](XXX)

### Ensemble-based Learning
*   [XXX 2024] XXX [paper](XXX)

### Large Time-Series Models

*   [XXX 2024] XXX [paper](XXX)

## OOD evaluation

*   [XXX 2024] XXX [paper](XXX)


## Other Related Papers

*   [XXX 2024] XXX [paper](XXX)


## Dataset

*   [XXX 2024] XXX [paper](XXX)

#### Acknowledgement

Last updated on March 6, 2025. (For problems, contact xinwu5386@gmail.com. To add papers, please pull request at <a href="https://github.com/tsood-generalization/tsood-generalization.github.io">our repo</a>)

<div style="width: 200px; height: 150px; margin: 0 auto;">
<!-- Map Widget -->
<!-- <script type="text/javascript" id="clustrmaps" src="//clustrmaps.com/map_v2.js?d=q6eVgeaBn-p2jkFoYf-6vSskb8SxHJqWuia9GW0Q_AE&cl=ffffff&w=a"></script> -->
<!-- Globe Widget -->
  <script type="text/javascript" id="clstr_globe" src="//clustrmaps.com/globe.js?d=q6eVgeaBn-p2jkFoYf-6vSskb8SxHJqWuia9GW0Q_AE"></script>
</div>
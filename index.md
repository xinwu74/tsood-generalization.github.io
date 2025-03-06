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

*  <span style="color:red;">[CVPR 2021]</span> Out-of-distribution Detection and Generation using Soft Brownian Offset Sampling and Autoencoders <a href="https://openaccess.thecvf.com/content/CVPR2021W/SAIAD/papers/Moller_Out-of-Distribution_Detection_and_Generation_Using_Soft_Brownian_Offset_Sampling_and_CVPRW_2021_paper.pdf">[paper]</a>
*   <span style="color:red;">[ICLR 2022]</span> Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift <a href="https://openreview.net/pdf?id=cGDAkQo1C0p">[paper]</a>
*   <span style="color:red;">[SMC 2022]</span> Feature Importance Identification for Time Series Classifiers <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9945205">[paper]</a>
*   <span style="color:red;">[FUZZ 2023]</span> An Initial Step Towards Stable Explanations for Multivariate Time Series Classifiers with LIME<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10309814">[paper]</a>
*   <span style="color:red;">[Inf. Sci. 2023]</span> Explaining time series classifiers through meaningful perturbation and optimisation <a href="https://www.sciencedirect.com/science/article/pii/S0020025523009192">[paper]</a>
*   <span style="color:red;">[Neural Networks 2024]</span> SEGAL time series classification - Stable explanations using a generative model and an adaptive weighting method for LIME <a href="https://www.sciencedirect.com/science/article/pii/S0893608024002697/pdfft?md5=3f81e6d7a6bddcb6857d94aa6ab04937&pid=1-s2.0-S0893608024002697-main.pdf">[paper]</a>
*   <span style="color:red;">[VR 2024]</span> Generating Virtual Reality Stroke Gesture Data from Out-of-Distribution Desktop Stroke Gesture Data <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10494175">[paper]</a>
*   <span style="color:red;">[AAAI 2024]</span> Generalizing across temporal domains with koopman operators <a href="https://ojs.aaai.org/index.php/AAAI/article/view/29604/31020">[paper]</a>
*   <span style="color:red;">[IJCAI 2024]</span> Temporal Domain Generalization via Learning Instance-level Evolving Patterns <a href="https://www.ijcai.org/proceedings/2024/0470.pdf">[paper]</a>
*   <span style="color:red;">[ICML 2024]</span> Connect later: Improving fine-tuning for robustness with targeted augmentations <a href=" https://openreview.net/pdf?id=Uz4Qr40Y3C">[paper]</a>
*   <span style="color:red;">[ICML 2024]</span> TimeX++: Learning Time-Series Explanations with Information Bottleneck<a href="https://openreview.net/pdf?id=t6dBpwkbea">[paper]</a>

### Concept Shift

*   <span style="color:red;">[AAMAS 2024]</span> Rethinking out-of-distribution detection for reinforcement learning: Advancing methods for evaluation and detection, <a href="https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1445.pdf">[paper]</a>
*  <span style="color:red;">[ICLR 2024]</span> Disentangling Time Series Representations via Contrastive Independence-of-Support on l-Variational Inference <a href="https://openreview.net/pdf?id=iI7hZSczxE">[paper]</a>

## Representation learning


### Decoupling-based Methods
Multi-Structured Analysis:
*   <span style="color:red;">[NIPS 2020]</span> Feature Shift Detection: Localizing Which Features Have Shifted via Conditional Distribution Tests<a href="https://proceedings.neurips.cc/paper/2020/file/e2d52448d36918c575fa79d88647ba66-Paper.pdf">[paper]</a>
*   <span style="color:red;">[IJCNN 2021]</span> Unsupervised Energy-based Out-of-distribution Detection using Stiefel-Restricted Kernel Machine<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9533706">[paper]</a>
*   <span style="color:red;">[ACM TCPS 2022]</span> Efficient Out-of-Distribution Detection Using Latent Space of β-VAE for Cyber-Physical Systems<a href="https://dl.acm.org/doi/pdf/10.1145/3491243">[paper]</a>
*   <span style="color:red;">[ACM TIST 2023]</span> Out-of-distribution Detection in Time-series Domain: A Novel Seasonal Ratio Scoring Approach<a href="https://dl.acm.org/doi/pdf/10.1145/3630633">[paper]</a>
*   <span style="color:red;">[KDD 2024]</span> Orthogonality Matters: Invariant Time Series Representation for Out-of-distribution Classification<a href="https://dl.acm.org/doi/pdf/10.1145/3637528.3671768">[paper]</a>
*   <span style="color:red;">[TKDE 2024]</span> Disentangling Structured Components: Towards Adaptive, Interpretable and Scalable Time Series Forecasting <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10457027">[paper]</a>
*   <span style="color:red;">[AAAI 2024]</span> MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting<a href="https://ojs.aaai.org/index.php/AAAI/article/view/28991/29883">[paper]</a>

Causality-Inspired:
*   <span style="color:red;">[CVPR 2021]</span> Causal hidden markov model for time series disease forecasting<a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Causal_Hidden_Markov_Model_for_Time_Series_Disease_Forecasting_CVPR_2021_paper.pdf">[paper]</a>
*   <span style="color:red;">[ICRA 2022]</span> Causal-based Time Series Domain Generalization for Vehicle Intention Prediction <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9812188">[paper]</a>
*   <span style="color:red;">[ICML 2023]</span> Neural Stochastic Differential Games for Time-series Analysis <a href="https://proceedings.mlr.press/v202/park23j/park23j.pdf">[paper]</a>
*   <span style="color:red;">[Sci. Robot. 2023]</span> Robust flight navigation out of distribution with liquid
neural networks <a href="https://cap.csail.mit.edu/sites/default/files/research-pdfs/Robust%20flight%20navigation%20out%20of%20distribution%20with%20liquid%20neural%20networks.pdf">[paper]</a>
*   <span style="color:red;">[Inf. Sci. 2024]</span> A causal representation learning based model for time series prediction under external interference  <a href="https://www.sciencedirect.com/science/article/abs/pii/S002002552400183X">[paper]</a>


### Invariant-based Methods
Invariant Risk Minimization:
*   <span style="color:red;">[AAAI 2021]</span> Meta-learning framework with applications to zero-shot time-series forecasting <a href="https://ojs.aaai.org/index.php/AAAI/article/view/17115/16922">[paper]</a>
*   <span style="color:red;">[ICML 2024]</span> Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning<a href="https://openreview.net/pdf?id=SMUXPVKUBg">[paper]</a>
*   <span style="color:red;">[ICRA 2023]</span> Robust Forecasting for Robotic Control: A Game-Theoretic Approach <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10160721">[paper]</a>
*   <span style="color:red;">[KDD 2023]</span> DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting<a href="https://dl.acm.org/doi/pdf/10.1145/3580305.3599315">[paper]</a>
*   <span style="color:red;">[KDD 2023]</span> TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting <a href="https://dl.acm.org/doi/pdf/10.1145/3580305.3599533">[paper]</a>
*   <span style="color:red;">[WWW 2024]</span> Towards Invariant Time Series Forecasting in Smart Cities <a href="https://dl.acm.org/doi/proceedings/10.1145/3589335?tocHeading=heading10">[paper]</a>


Domain-Invariance:
*   <span style="color:red;">[ICLR 2021]</span> In-N-Out: Pre-Training and Self-Training using Auxiliary Information for Out-of-Distribution Robustness<a href="https://openreview.net/pdf?id=jznizqvr15J">[paper]</a>
*   <span style="color:red;">[NIPS 2023]</span> Evolving Standardization for Continual Domain Generalization over Temporal Drift<a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/459a911eb49cd2e0192055ee156d04e5-Paper-Conference.pdf">[paper]</a>
*   <span style="color:red;">[ICLR 2023]</span> Out-of-distribution Representation Learning for Time Series Classification <a href="https://openreview.net/pdf?id=gUZWOE42l6Q">[paper]</a>
*   <span style="color:red;">[VR 2024]</span> Generating Virtual Reality Stroke Gesture Data from Out-of-Distribution Desktop Stroke Gesture Data <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10494175">[paper]</a>
*   <span style="color:red;">[AAAI 2024]</span> Generalizing across temporal domains with koopman operators<a href="https://ojs.aaai.org/index.php/AAAI/article/view/29604/31020">[paper]</a>
*   <span style="color:red;">[ICML 2024]</span> Connect later: Improving fine-tuning for robustness with targeted augmentations<a href="https://openreview.net/pdf?id=Uz4Qr40Y3C">[paper]</a>
*   <span style="color:red;">[NIPS 2024]</span> Continuous Temporal Domain Generalization <a href="https://openreview.net/pdf?id=G24fOpC3JE">[paper]</a>
*   <span style="color:red;">[TPAMI 2024]</span> Diversify: A General Framework for Time Series Out-of-Distribution Detection and Generalization <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10402053">[paper]</a>
*   <span style="color:red;">[Struct. 2024]</span> Enhancing time series data classification for structural damage detection through out-of-distribution representation learning<a href="https://www.sciencedirect.com/science/article/abs/pii/S2352012424009184">[paper]</a>
*   <span style="color:red;">[DAC 2024]</span> SMORE: Similarity-based Hyperdimensional Domain Adaptation for Multi-Sensor Time Series Classification<a href="https://dl.acm.org/doi/pdf/10.1145/3649329.3658477">[paper]</a> <a href="https://bpb-us-e2.wpmucdn.com/sites.uci.edu/dist/9/5133/files/2024/07/DAC_2024.pdf">[PPT]</a>


### Ensemble-based Learning
*   <span style="color:red;">[SAIS 2022]</span> Out-of-distribution in Human Activity Recognition <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9833052">[paper]</a>
*   <span style="color:red;">[Reliab. Eng. Syst. Saf 2022]</span> Out-of-distribution detection-assisted trustworthy machinery fault diagnosis approach with uncertainty-aware deep ensembles <a href="https://www.sciencedirect.com/science/article/abs/pii/S0951832022002836">[paper]</a>
*   <span style="color:red;">[KDD 2023]</span> Maintaining the Status Quo: Capturing Invariant Relations for OOD Spatiotemporal Learning <a href="http://home.ustc.edu.cn/~zzy0929/Home/Paper/KDD23_CauSTG.pdf">[paper]</a>
*   <span style="color:red;">[ICC 2023]</span> Out-of-distribution Internet Traffic Prediction Generalization Using Deep Sequence Model <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10279740">[paper]</a>
*   <span style="color:red;">[AI Commun. 2023]</span> Classifying falls using out-of-distribution detection in human activity recognition <a href="https://content.iospress.com/download/ai-communications/aic220205?id=ai-communications%2Faic220205">[paper]</a>
*   <span style="color:red;">[AAMAS 2024]</span> Rethinking out-of-distribution detection for reinforcement learning: Advancing methods for evaluation and detection, <a href="https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1445.pdf">[paper]</a>

### Large Time-Series Models

*   <span style="color:red;">[XXX 2024]</span> XXX <a href=" ">[paper]</a>

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
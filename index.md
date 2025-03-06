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

*  <span style="color:red;">[CVPR 2021]</span> Out-of-distribution Detection and Generation using Soft Brownian Offset Sampling and Autoencoders <a href="https://openaccess.thecvf.com/content/CVPR2021W/SAIAD/papers/Moller_Out-of-Distribution_Detection_and_Generation_Using_Soft_Brownian_Offset_Sampling_and_CVPRW_2021_paper.pdf">[Paper]</a>
*   <span style="color:red;">[ICLR 2022]</span> Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift <a href="https://openreview.net/pdf?id=cGDAkQo1C0p">[Paper]</a>
*   <span style="color:red;">[SMC 2022]</span> Feature Importance Identification for Time Series Classifiers <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9945205">[Paper]</a>
*   <span style="color:red;">[FUZZ 2023]</span> An Initial Step Towards Stable Explanations for Multivariate Time Series Classifiers with LIME <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10309814">[Paper]</a>
*   <span style="color:red;">[Inf. Sci. 2023]</span> Explaining time series classifiers through meaningful perturbation and optimisation <a href="https://www.sciencedirect.com/science/article/pii/S0020025523009192">[Paper]</a>
*   <span style="color:red;">[J.Neunet 2024]</span> SEGAL time series classification - Stable explanations using a generative model and an adaptive weighting method for LIME <a href="https://www.sciencedirect.com/science/article/pii/S0893608024002697/pdfft?md5=3f81e6d7a6bddcb6857d94aa6ab04937&pid=1-s2.0-S0893608024002697-main.pdf">[Paper]</a>
*   <span style="color:red;">[VR 2024]</span> Generating Virtual Reality Stroke Gesture Data from Out-of-Distribution Desktop Stroke Gesture Data <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10494175">[Paper]</a>
*   <span style="color:red;">[AAAI 2024]</span> Generalizing across temporal domains with koopman operators <a href="https://ojs.aaai.org/index.php/AAAI/article/view/29604/31020">[Paper]</a>
*   <span style="color:red;">[IJCAI 2024]</span> Temporal Domain Generalization via Learning Instance-level Evolving Patterns <a href="https://www.ijcai.org/proceedings/2024/0470.pdf">[Paper]</a>
*   <span style="color:red;">[ICML 2024]</span> Connect later: Improving fine-tuning for robustness with targeted augmentations <a href=" https://openreview.net/pdf?id=Uz4Qr40Y3C">[Paper]</a>
*   <span style="color:red;">[ICML 2024]</span> TimeX++: Learning Time-Series Explanations with Information Bottleneck <a href="https://openreview.net/pdf?id=t6dBpwkbea">[Paper]</a>

### Concept Shift

*   <span style="color:red;">[AAMAS 2024]</span> Rethinking out-of-distribution detection for reinforcement learning: Advancing methods for evaluation and detection <a href="https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1445.pdf">[Paper]</a>
*  <span style="color:red;">[ICLR 2024]</span> Disentangling Time Series Representations via Contrastive Independence-of-Support on l-Variational Inference <a href="https://openreview.net/pdf?id=iI7hZSczxE">[Paper]</a>

## Representation learning


### Decoupling-based Methods
Multi-Structured Analysis:
*   <span style="color:red;">[NIPS 2020]</span> Feature Shift Detection: Localizing Which Features Have Shifted via Conditional Distribution Tests <a href="https://proceedings.neurips.cc/paper/2020/file/e2d52448d36918c575fa79d88647ba66-Paper.pdf">[Paper]</a>
*   <span style="color:red;">[IJCNN 2021]</span> Unsupervised Energy-based Out-of-distribution Detection using Stiefel-Restricted Kernel Machine <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9533706">[Paper]</a>
*   <span style="color:red;">[ACM TCPS 2022]</span> Efficient Out-of-Distribution Detection Using Latent Space of β-VAE for Cyber-Physical Systems <a href="https://dl.acm.org/doi/pdf/10.1145/3491243">[Paper]</a>
*   <span style="color:red;">[ACM TIST 2023]</span> Out-of-distribution Detection in Time-series Domain: A Novel Seasonal Ratio Scoring Approach <a href="https://dl.acm.org/doi/pdf/10.1145/3630633">[Paper]</a>
*   <span style="color:red;">[KDD 2024]</span> Orthogonality Matters: Invariant Time Series Representation for Out-of-distribution Classification <a href="https://dl.acm.org/doi/pdf/10.1145/3637528.3671768">[Paper]</a>
*   <span style="color:red;">[TKDE 2024]</span> Disentangling Structured Components: Towards Adaptive, Interpretable and Scalable Time Series Forecasting <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10457027">[Paper]</a>
*   <span style="color:red;">[AAAI 2024]</span> MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting <a href="https://ojs.aaai.org/index.php/AAAI/article/view/28991/29883">[Paper]</a>

Causality-Inspired:
*   <span style="color:red;">[CVPR 2021]</span> Causal hidden markov model for time series disease forecasting <a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Causal_Hidden_Markov_Model_for_Time_Series_Disease_Forecasting_CVPR_2021_paper.pdf">[Paper]</a>
*   <span style="color:red;">[ICRA 2022]</span> Causal-based Time Series Domain Generalization for Vehicle Intention Prediction <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9812188">[Paper]</a>
*   <span style="color:red;">[ICML 2023]</span> Neural Stochastic Differential Games for Time-series Analysis <a href="https://proceedings.mlr.press/v202/park23j/park23j.pdf">[Paper]</a>
*   <span style="color:red;">[Sci. Robot. 2023]</span> Robust flight navigation out of distribution with liquid
neural networks <a href="https://cap.csail.mit.edu/sites/default/files/research-pdfs/Robust%20flight%20navigation%20out%20of%20distribution%20with%20liquid%20neural%20networks.pdf">[Paper]</a>
*   <span style="color:red;">[Inf. Sci. 2024]</span> A causal representation learning based model for time series prediction under external interference  <a href="https://www.sciencedirect.com/science/article/abs/pii/S002002552400183X">[Paper]</a>


### Invariant-based Methods
Invariant Risk Minimization Models:
*   <span style="color:red;">[AAAI 2021]</span> Meta-learning framework with applications to zero-shot time-series forecasting <a href="https://ojs.aaai.org/index.php/AAAI/article/view/17115/16922">[Paper]</a>
*   <span style="color:red;">[ICML 2024]</span> Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning <a href="https://openreview.net/pdf?id=SMUXPVKUBg">[Paper]</a>
*   <span style="color:red;">[ICRA 2023]</span> Robust Forecasting for Robotic Control: A Game-Theoretic Approach <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10160721">[Paper]</a>
*   <span style="color:red;">[KDD 2023]</span> DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting <a href="https://dl.acm.org/doi/pdf/10.1145/3580305.3599315">[Paper]</a>
*   <span style="color:red;">[KDD 2023]</span> TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting <a href="https://dl.acm.org/doi/pdf/10.1145/3580305.3599533">[Paper]</a>
*   <span style="color:red;">[WWW 2024]</span> Towards Invariant Time Series Forecasting in Smart Cities <a href="https://dl.acm.org/doi/proceedings/10.1145/3589335?tocHeading=heading10">[Paper]</a>


Domain-Invariance Methods:
*   <span style="color:red;">[ICLR 2021]</span> In-N-Out: Pre-Training and Self-Training using Auxiliary Information for Out-of-Distribution Robustness <a href="https://openreview.net/pdf?id=jznizqvr15J">[Paper]</a>
*   <span style="color:red;">[NIPS 2023]</span> Evolving Standardization for Continual Domain Generalization over Temporal Drift <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/459a911eb49cd2e0192055ee156d04e5-Paper-Conference.pdf">[Paper]</a>
*   <span style="color:red;">[ICLR 2023]</span> Out-of-distribution Representation Learning for Time Series Classification <a href="https://openreview.net/pdf?id=gUZWOE42l6Q">[Paper]</a>
*   <span style="color:red;">[VR 2024]</span> Generating Virtual Reality Stroke Gesture Data from Out-of-Distribution Desktop Stroke Gesture Data <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10494175">[Paper]</a>
*   <span style="color:red;">[AAAI 2024]</span> Generalizing across temporal domains with koopman operators <a href="https://ojs.aaai.org/index.php/AAAI/article/view/29604/31020">[Paper]</a>
*   <span style="color:red;">[ICML 2024]</span> Connect later: Improving fine-tuning for robustness with targeted augmentations <a href="https://openreview.net/pdf?id=Uz4Qr40Y3C">[Paper]</a>
*   <span style="color:red;">[NIPS 2024]</span> Continuous Temporal Domain Generalization <a href="https://openreview.net/pdf?id=G24fOpC3JE">[Paper]</a>
*   <span style="color:red;">[TPAMI 2024]</span> Diversify: A General Framework for Time Series Out-of-Distribution Detection and Generalization <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10402053">[Paper]</a>
*   <span style="color:red;">[Struct. 2024]</span> Enhancing time series data classification for structural damage detection through out-of-distribution representation learning <a href="https://www.sciencedirect.com/science/article/abs/pii/S2352012424009184">[Paper]</a>
*   <span style="color:red;">[DAC 2024]</span> SMORE: Similarity-based Hyperdimensional Domain Adaptation for Multi-Sensor Time Series Classification <a href="https://dl.acm.org/doi/pdf/10.1145/3649329.3658477">[Paper]</a> <a href="https://bpb-us-e2.wpmucdn.com/sites.uci.edu/dist/9/5133/files/2024/07/DAC_2024.pdf">[PPT]</a>


### Ensemble-based Learning
*   <span style="color:red;">[SAIS 2022]</span> Out-of-distribution in Human Activity Recognition <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9833052">[Paper]</a>
*   <span style="color:red;">[RESS 2022]</span> Out-of-distribution detection-assisted trustworthy machinery fault diagnosis approach with uncertainty-aware deep ensembles <a href="https://www.sciencedirect.com/science/article/abs/pii/S0951832022002836">[Paper]</a>
*   <span style="color:red;">[KDD 2023]</span> Maintaining the Status Quo: Capturing Invariant Relations for OOD Spatiotemporal Learning <a href="http://home.ustc.edu.cn/~zzy0929/Home/Paper/KDD23_CauSTG.pdf">[Paper]</a>
*   <span style="color:red;">[ICC 2023]</span> Out-of-distribution Internet Traffic Prediction Generalization Using Deep Sequence Model <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10279740">[Paper]</a>
*   <span style="color:red;">[AIC 2023]</span> Classifying falls using out-of-distribution detection in human activity recognition <a href="https://content.iospress.com/download/ai-communications/aic220205?id=ai-communications%2Faic220205">[Paper]</a>
*   <span style="color:red;">[AAMAS 2024]</span> Rethinking out-of-distribution detection for reinforcement learning: Advancing methods for evaluation and detection <a href="https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1445.pdf">[Paper]</a>

### Large Time-Series Models

Tuning-based Methods:
*   <span style="color:red;">[NIPS 2023]</span> ForecastPFN: Synthetically-Trained Zero-Shot Forecasting <a href="https://openreview.net/pdf?id=tScBQRNgjk">[Paper]</a>
*   <span style="color:red;">[NIPS 2024]</span> Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-shot Forecasting of Multivariate Time Series <a href="https://openreview.net/pdf?id=3O5YCEWETq">[Paper]</a>
*   <span style="color:red;">[ICASSP 2024]</span> ETP: Learning Transferable ECG Representations via ECG-Text Pre-Training <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10446742">[Paper]</a>
*   <span style="color:red;">[NIPS 2023]</span> JoLT: Jointly Learned Representations of
Language and Time-Series <a href="https://openreview.net/pdf?id=UVF1AMBj9u">[Paper]</a>
*   <span style="color:red;">[AAAI 2024]</span> JoLT: Jointly Learned Representations of Language and Time-Series for Clinical Time-Series Interpretation (Student Abstract) <a href="https://ojs.aaai.org/index.php/AAAI/article/view/30423/32496">[Paper]</a>
*   <span style="color:red;">[ACL 2023]</span> Transfer Knowledge from Natural Language to Electrocardiography: Can We Detect Cardiovascular Disease Through Language Models? <a href="https://aclanthology.org/2023.findings-eacl.33.pdf">[Paper]</a>
*   <span style="color:red;">[ICLR 2024]</span> Time-LLM: Time Series Forecasting by Reprogramming Large Language Models <a href="https://openreview.net/pdf?id=Unb5CVPtae">[Paper]</a>
*   <span style="color:red;">[AAAI 2025]</span> CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning <a href="https://github.com/Hank0626/CALF">[Code]</a>
*   <span style="color:red;">[ICML 2024]</span> Unified training of universal time series forecasting transformers <a href="https://dl.acm.org/doi/10.5555/3692070.3694248">[Paper]</a> <a href="https://github.com/SalesforceAIResearch/uni2ts">[Code]</a>  
*   <span style="color:red;">[CIKM 2024]</span> General Time Transformer: an Encoder-only Foundation Model for Zero-Shot Multivariate Time Series Forecasting <a href="https://dl.acm.org/doi/pdf/10.1145/3627673.3679931">[Paper]</a>
*   <span style="color:red;">[NIPS 2024]</span> Align and Fine-Tune: Enhancing LLMs for Time-Series Forecasting <a href="https://openreview.net/pdf?id=AaRCmJieG4">[Paper]</a>
*   <span style="color:red;">[AAAI 2025]</span> ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data <a href="https://github.com/forestsking/chattime">[Code]</a>

Non-tuning-based Methods:
*   <span style="color:red;">[NIPS 2023]</span> Large Language Models Are Zero-Shot Time Series Forecasters <a href="https://openreview.net/pdf?id=md68e8iZK1">[Paper]</a>
*   <span style="color:red;">[ArXiv 2023]</span> TimeGPT-1 <a href="https://arxiv.org/abs/2310.03589">[Paper]</a> <a href="https://github.com/Nixtla/nixtla">[Code]</a>
*   <span style="color:red;">[ArXiv 2024]</span> TableTime: Reformulating Time Series Classification as Training-Free Table Understanding with Large Language Models <a href="https://arxiv.org/abs/2411.15737">[Paper]</a> <a href="https://github.com/realwangjiahao/tabletime">[Code]</a>
*   <span style="color:red;">[ArXiv 2023]</span> Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain <a href="https://arxiv.org/abs/2310.05063">[Paper]</a> <a href="https://github.com/SalesforceAIResearch/pretrain-time-series-cloudops">[Code]</a>

Others:
*   <span style="color:red;">[NIPS 2023]</span> One Fits All: Power General Time Series Analysis by Pretrained LM <a href="https://openreview.net/pdf?id=gMS6FVZvmF">[Paper]</a>
*   <span style="color:red;">[R0-FoMo 2023]</span> Lag-Llama: Towards Foundation Models for Time Series Forecasting <a href="https://openreview.net/pdf?id=jYluzCLFDM">[Paper]</a>
*   <span style="color:red;">[ICML 2024]</span> MOMENT: A Family of Open Time-series Foundation Models <a href="file:///Users/xinwu/Downloads/goswami24a.pdf">[Paper]</a> <a href="https://github.com/moment-timeseries-foundation-model/moment">[Code]</a>
*   <span style="color:red;">[ICML 2024]</span> Timer: Generative Pre-trained Transformers Are Large Time Series Models <a href="https://dl.acm.org/doi/10.5555/3692070.3693383">[Paper]</a> <a href="https://github.com/thuml/Large-Time-Series-Model">[Code]</a>
*   <span style="color:red;">[ICML 2024]</span> A Decoder-only Foundation Model for Time-series Forecasting <a href="https://raw.githubusercontent.com/mlresearch/v235/main/assets/das24c/das24c.pdf">[Paper]</a> <a href="https://github.com/google-research/timesfm?tab=readme-ov-file">[Code]</a>
*   <span style="color:red;">[TMLR 2024]</span> Chronos: Learning the Language of Time Series <a href="https://openreview.net/pdf?id=gerNCVqqtR">[Paper]</a> <a href="https://github.com/amazon-science/chronos-forecasting">[Code]</a>


## Datasets

| Paper        | Datasets          | 
|:-------------|:------------------|
| [SLIME-MTS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10309814)           | [UEA](https://www.timeseriesclassification.com/index.php) |
| [Causal-HMM](https://github.com/LilJing/causal_hmm) | [In-house data on PPA](https://github.com/LilJing/causal_hmm) |
| [TimeX++](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) | [UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) |
| [CauSTG](https://data.cic-tp.com/h5/sample-data/china/export-data/company/suzhou-industrial-park) | [SIP]({https://data.cic-tp.com/h5/sample-data/china/export-data/company/suzhou-industrial-park), [METR-LA](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset), [KnowAir](https://github.com/shuowang-ai/PM2.5-GNN), [Electricity](https://github.com/laiguokun/multivariate-time-series-data/tree/master/electricity) |
| [SCNN](https://github.com/laiguokun/multivariate-time-series-data) | [Traffic, Solar-energy, Electricity, Exchange-rate](https://github.com/laiguokun/multivariate-time-series-data) |
| [MSGNet](https://opensky-network.org/) | [Flight](https://opensky-network.org) |
| [MSGNet](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) | [Weather, ETT](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) |
| [MSGNet](https://github.com/laiguokun/multivariate-time-series-data/tree/master/electricity) | [Exchange-Rate, Electricity](https://github.com/laiguokun/multivariate-time-series-data/tree/master/electricity) |



## Other Related Papers

*   <span style="color:red;">[ArXiv 2021]</span> Towards out-of-distribution generalization: A survey <a href="https://arxiv.org/pdf/2108.13624">[Paper]</a>
*   <span style="color:red;">[ArXiv 2022]</span>Out-of-distribution generalization on graphs: A survey<a href="https://arxiv.org/pdf/2202.07987">[Paper]</a>
*   <span style="color:red;">[TPAMI 2022]</span>Domain generalization: A survey<a href="">[Paper]</a>
*   <span style="color:red;">[TKDE 2023]</span>Generalizing to unseen domains: A survey on domain generalization<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9782500">[Paper]</a>
*   <span style="color:red;">[ACM Comput. Surv. 2023]</span> Generative adversarial networks in time series: A systematic literature review<a href="https://dl.acm.org/doi/pdf/10.1145/3559540">[Paper]</a>
*   <span style="color:red;">[TPAMI 2024]</span> Self-supervised learning for time series analysis: Taxonomy, progress, and prospects<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10496248">[Paper]</a>
*   <span style="color:red;">[TPAMI 2024]</span> A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10636792">[Paper]</a>
*   <span style="color:red;">[TKDE 2024]</span>A survey on time-series pre-trained
models<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10706809">[Paper]</a>
*   <span style="color:red;">[IJCV 2024]</span> Generalized out-of-distribution detection: A survey <a href="https://link.springer.com/article/10.1007/s11263-024-02117-4">[Paper]</a>


#### Acknowledgement

Last updated on March 6, 2025. (For problems, contact xinwu5386@gmail.com. To add papers, please pull request at <a href="https://github.com/tsood-generalization/tsood-generalization.github.io">our repo</a>)

<div style="width: 200px; height: 150px; margin: 0 auto;">
<!-- Map Widget -->
<!-- <script type="text/javascript" id="clustrmaps" src="//clustrmaps.com/map_v2.js?d=q6eVgeaBn-p2jkFoYf-6vSskb8SxHJqWuia9GW0Q_AE&cl=ffffff&w=a"></script> -->
<!-- Globe Widget -->
  <script type="text/javascript" id="clstr_globe" src="//clustrmaps.com/globe.js?d=q6eVgeaBn-p2jkFoYf-6vSskb8SxHJqWuia9GW0Q_AE"></script>
</div>
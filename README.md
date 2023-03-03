# Explainable Human-centered Traits from Head Motion and Facial Expression Dynamics

## Abstract: 
We explore the efficacy of multimodal behavioral cues for explainable prediction of personality and interview-specific traits. We utilize elementary head-motion units named kinemes, atomic facial movements termed action units and speech features to estimate these human-centered traits. Empirical results confirm that kinemes and action units enable discovery of multiple trait-specific behaviors while also enabling explainability in support of the predictions. For fusing cues, we explore decision and feature-level fusion, and an additive attention-based fusion strategy which quantifies the relative importance of the three modalities for trait prediction. Examining various long-short term memory (LSTM) architectures for classification and regression on the MIT Interview and First Impressions Candidate Screening (FICS) datasets, we note that: (1) Multimodal approaches outperform unimodal counterparts; (2) Efficient trait predictions and plausible explanations are achieved with both unimodal and multimodal approaches, and (3) Following the thin-slice approach, effective trait prediction is achieved even from two-second behavioral snippets.

## Framework Overview:
![framework_overview](https://user-images.githubusercontent.com/79365852/222646735-01cce1ca-7b8c-4191-a8df-1848e83a4c58.jpg)
*Overview of the proposed framework: Kinemes (elementary head motions), action units (atomic facial movements) and speech features employed for explainable trait prediction*

## Implementation Details:
**Preprocessing** -- We've used openface(https://github.com/TadasBaltrusaitis/OpenFace) to extract pose angles and action units for kineme and AU generation. For audio feature generation, we've used Librosa(https://librosa.org/doc/latest/index.html) python package.

**Model Architecture** -- We've done Chunk level and video level analysis for both the datasets MIT and FICS. For more details, please refer to paper.

## Related Links:
[Paper Link](https://arxiv.org/pdf/2302.09817v2.pdf)

[Head Matters Paper (ICMI 2021)](https://dl.acm.org/doi/10.1145/3462244.3479901)

[Head Matters Code](https://github.com/MonikaGahalawat11/Head-Matters--Code) 



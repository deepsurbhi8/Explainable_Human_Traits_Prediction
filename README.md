# Explainable Human-centered Traits from Head Motion and Facial Expression Dynamics

## Abstract: 
We explore the efficacy of multimodal behavioral cues for explainable prediction of personality and interview-specific traits. We utilize elementary head-motion units named kinemes, atomic facial movements termed action units and speech features to estimate these human-centered traits. Empirical results confirm that kinemes and action units enable discovery of multiple trait-specific behaviors while also enabling explainability in support of the predictions. For fusing cues, we explore decision and feature-level fusion, and an additive attention-based fusion strategy which quantifies the relative importance of the three modalities for trait prediction. Examining various long-short term memory (LSTM) architectures for classification and regression on the MIT Interview and First Impressions Candidate Screening (FICS) datasets, we note that: (1) Multimodal approaches outperform unimodal counterparts; (2) Efficient trait predictions and plausible explanations are achieved with both unimodal and multimodal approaches, and (3) Following the thin-slice approach, effective trait prediction is achieved even from two-second behavioral snippets.

## Framework Overview:
![framework_overview](https://github.com/deepsurbhi8/Explainable_Human_Traits_Prediction/assets/79365852/9c3480ab-78c1-40ae-89bb-02b75503fa37)
*Overview of the proposed framework: Kinemes (elementary head motions), action units (atomic facial movements) and speech features employed for explainable trait prediction*

## Implementation Details:
**Preprocessing** -- We've used [Openface](https://github.com/TadasBaltrusaitis/OpenFace) to extract pose angles and action units for kineme and AU generation. For audio feature generation, we've used [Librosa](https://librosa.org/doc/latest/index.html) python package.

**Model Architecture** -- We've done Chunk level and video level analysis for both the datasets MIT and FICS. For more details, please refer to the paper.

**Requirements** -- This code was tested on Python 3.8, Tensorflow 2.7 and Keras 2.7. It is recommended to use the appropraite versions of each library to implement the code.

## Repository Structure:
 .
    ├── README.md    
    ├── Architectures           
    │   ├── Attention_fusion_figure.pdf      # Additive attention fusion architecture overview
    │   ├── Framework_overview.pdf           # Overview of the proposed framework
    │   └── LSTM_arch.pdf                    # Trimodal feature fusion architecture
    ├── Codes         
    │   ├── Bimodal Fusion                   # Bimodal implementation of different combination of features (AU, Kineme and Audio features)        
    │   │   ├── Classification_Feature_au_kin_VL_MIT.py  # (Eg: contains the code to implement video-level classification appraoch over the MIT dataset using AU and Kineme features using Feature-level fusion)
    │   ├── Feature Extraction               
    │   │   ├── Action_units_data_prep.py    # Code to create Action Unit data matrix from the openface extracted au files
    │   │   ├── Audio_chunk_formation.py     # Code to create chunks from the Audio data matrix 
    │   │   ├── Kineme_data_prep.py          # Code to create kineme feature data matrix for train and test set files
    │   ├── Trimodal Fusion                  # Different approaches (Decision, Feature and Attention-based fusion) of the three modalities (AU, Kineme, Audio features)
    │   ├── Unimodal Approach                # Single modality code implementation over the two datasets
    ├── Data         
    │   ├── FICS_dataset                  # Bimodal implementation of different combination of features (AU, Kineme and Audio features)        
    │   │   ├── FICS_test_files.zip       # Features (Action Unit, Kineme and Audio) extracted over the test set of FICS dataset
    │   │   ├── FICS_train_files.zip      # Features (Action Unit, Kineme and Audio) extracted over the train set of FICS dataset
    │   │   ├── FICS_val_files.zip        # Features (Action Unit, Kineme and Audio) extracted over the validation set of FICS dataset
    │   ├── MIT_dataset    
    │   │   ├── MIT_AU_features.zip       # Action Unit features extracted over all files of the MIT dataset
    │   │   ├── MIT_kineme_features.zip   # Kineme representation for the head pose values over the MIT dataset
    ├── Presentation         
    │   ├── Explainable_Human_Traits_Prediction.pdf        


## Related Links:
[Paper Link](https://arxiv.org/pdf/2302.09817v2.pdf)

[Head Matters Paper (ICMI 2021)](https://dl.acm.org/doi/10.1145/3462244.3479901)

[Head Matters Code](https://github.com/MonikaGahalawat11/Head-Matters--Code) 



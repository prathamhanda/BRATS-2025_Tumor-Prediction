flowchart TD

    %% ================= SSA INPUT =================
    subgraph SSA_Input
        I1[T1 Native MRI]
        I2[T1 Contrast MRI]
        I3[T2 Weighted MRI]
        I4[T2 FLAIR MRI]
        I5[SSA Ground Truth Labels]
    end

    %% ================= PREPROCESSING =================
    subgraph Preprocessing
        P1[SSA Label Mapping]
        P2[Label 3 to 4 Conversion]
        P3[Intensity Normalization]
        P4[Isotropic Resampling 1mm]
        P5[Patch Extraction 128x128x128]
        P6[Data Quality Validation]
    end

    %% ================= SPLIT =================
    S1[Train Validation Split 80 20]

    %% ================= MODEL =================
    subgraph Model_Architecture
        M1[MONAI 3D UNet]
        M2[4 Input Channels]
        M3[Encoder 32 to 512]
        M4[Skip Connections]
        M5[Decoder 512 to 32]
        M6[4 Output Classes]
    end

    %% ================= TRANSFER LEARNING =================
    subgraph Transfer_Learning
        T1[Glioma Pretrained Weights]
        T2[SSA Feature Adaptation]
        T3[Feature Transfer]
        T4[Fine Tuning]
        T5[Domain Specific Learning]
    end

    %% ================= TRAINING =================
    subgraph GPU_Training
        G1[Mixed Precision]
        G2[Weighted Cross Entropy]
        G3[Adam Optimizer]
        G4[Learning Rate 0.001]
        G5[Batch Size 2]
        G6[25 Epochs]
    end

    %% ================= EVALUATION =================
    subgraph Evaluation
        E1[Dice Score 0.8857]
        E2[Per Class Analysis]
        E3[Clinical Threshold 0.70]
        E4[Research Target 0.80]
        E5[Statistical Validation]
    end

    %% ================= APPLICATION =================
    subgraph SSA_Clinical_Application
        C1[Automated Segmentation]
        C2[Treatment Planning]
        C3[Surgical Guidance]
        C4[Radiation Therapy]
        C5[Response Monitoring]
        C6[Healthcare Equity]
    end

    %% ================= CONNECTIONS =================
    I1 --> P1
    I2 --> P1
    I3 --> P1
    I4 --> P1
    I5 --> P2

    P1 --> P3 --> P4 --> P5 --> P6 --> S1
    P2 --> S1

    S1 --> M1 --> M2 --> M3 --> M4 --> M5 --> M6

    M6 --> T1 --> T2 --> T3 --> T4 --> T5

    T5 --> G1
    T5 --> G2

    G1 --> G3 --> G4 --> G6
    G2 --> G5 --> G6

    G6 --> E1 --> E2 --> E5
    E2 --> E3
    E2 --> E4

    E5 --> C1 --> C2
    C1 --> C3
    C2 --> C4 --> C5 --> C6

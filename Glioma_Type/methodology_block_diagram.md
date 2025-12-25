flowchart TD

    %% ================= INPUT =================
    subgraph Input_Data
        I1[T1 MRI]
        I2[T1ce MRI]
        I3[T2 MRI]
        I4[FLAIR MRI]
        I5[Ground Truth]
    end

    %% ================= PREPROCESSING =================
    subgraph Preprocessing
        P1[Skull Stripping]
        P2[Co Registration]
        P3[Normalization]
        P4[Patch Extraction  128x128x128]
    end

    %% ================= DATA SPLIT =================
    S1[Train Validation Split]

    %% ================= MODEL =================
    subgraph Model_Architecture
        M1[3D UNet]
        M2[Encoder Decoder]
        M3[Skip Connections]
        M4[4 Output Classes]
    end

    %% ================= TRAINING =================
    subgraph Training
        T1[Cross Entropy Loss]
        T2[Adam Optimizer]
        T3[LR 0.001]
        T4[50 Epochs]
    end

    %% ================= EVALUATION =================
    subgraph Evaluation
        E1[Dice Score]
        E2[IoU]
        E3[Hausdorff Distance]
        E4[Volume Estimation]
    end

    %% ================= APPLICATION =================
    subgraph Clinical_Application
        C1[Tumor Segmentation]
        C2[Treatment Planning]
        C3[Radiation Therapy]
        C4[Response Monitoring]
    end

    %% ================= CONNECTIONS =================
    I1 --> P1
    I2 --> P1
    I3 --> P1
    I4 --> P1

    P1 --> P2 --> P3 --> P4 --> S1
    I5 --> S1

    S1 --> M1
    M1 --> M2 --> M3 --> M4

    M4 --> T1
    M4 --> T2

    T1 --> T3 --> T4
    T2 --> T4

    T4 --> E1
    T4 --> E2

    E1 --> E3 --> E4

    E3 --> C1 --> C2
    E4 --> C3 --> C4

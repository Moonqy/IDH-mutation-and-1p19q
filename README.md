<!-- PROJECT LOGO -->
<br />
<div align="center">




  <h3 align="center">A Deep Learning Model with FFT-Based Frequency Analysis and Multiscale Fusion for Predicting IDH Mutation and 1p/19q Co-Deletion</h3>
  <img width="1057" height="476" alt="The structure of HyCoSNet" src="https://github.com/user-attachments/assets/ee44f051-75b1-492f-bf96-0e7a61981c19" />
  <p align="justify">
    Gliomas are highly lethal brain tumors, with Isocitrate Dehydrogenase (IDH) mutation and 1p/19q co-deletion recognized as critical biomarkers for prognosis and treatment response. Existing deep learning (DL) methods often underutilize the unique anatomical and physiological features of structural Magnetic Resonance Imaging (sMRI) and functional MRI (fMRI), and rarely integrate clinicopathological features. To address these limitations, we propose a novel multi-modal DL model that systematically leverages frequency-domain fMRI analysis and integrates clinicopathological features to enhance biomarker prediction. Our model employs a dual-branch architecture with cross-modal interactions, incorporating four key innovations: First, we utilize an optimized DenseNet to extract distinct modality-specific features from 3D fMRI and sMRI. Second, we apply Fourier frequency transformations with the fMRI branch to capture global neural activity patterns. Third, we introduce a Multiscale Fusion Module that employs attention-based mechanisms to fuse complementary features from both modalities across multiple scales robustly. Finally, we integrate clinicopathological features via a dedicated fusion pathway, enriching the model with clinical context for precise biomarker prediction. Evaluated on the UCSF-PDGM dataset, our model achieved AUCs of 0.9779 and 0.9662 for IDH mutation and 1p/19q co-deletion prediction. Moreover, we achieved an AUC of 0.9739 across multiple datasets, confirming our method’s robustness. We employed SHapley Additive exPlanations to quantify feature contributions, revealing patient’s age significantly impacts IDH mutation prediction, while IDH mutation itself dominates 1p/19q co-deletion prediction. These findings underscore the superior predictive performance and adaptability of our approach, demonstrating multi-modal DL’s transformative potential for glioma molecular subtyping. 
    <br />
    <br />
    <br />
  </p>
</div>

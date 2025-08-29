<!-- PROJECT LOGO -->
<br />
<div align="center">




  <h3 align="center">A Deep Learning Model with FFT-Based Frequency Analysis and Multiscale Fusion for Predicting IDH Mutation and 1p/19q Co-Deletion</h3>
  <img width="653" height="383" alt="Screenshot 2025-08-29 at 9 54 40 AM" src="https://github.com/user-attachments/assets/d1526d37-e493-4bc9-828c-530fb3e6d6d2" />
  <p align="justify">
    Gliomas are highly lethal brain tumors, with Isocitrate Dehydrogenase (IDH) mutation and 1p/19q co-deletion recognized as critical biomarkers for prognosis and treatment response. Existing deep learning (DL) methods often underutilize the unique anatomical and physiological features of structural Magnetic Resonance Imaging (sMRI) and functional MRI (fMRI), and rarely integrate clinicopathological features. To address these limitations, we propose a novel multi-modal DL model that systematically leverages frequency-domain fMRI analysis and integrates clinicopathological features to enhance biomarker prediction. Our model employs a dual-branch architecture with cross-modal interactions, incorporating four key innovations: First, we utilize an optimized DenseNet to extract distinct modality-specific features from 3D fMRI and sMRI. Second, we apply Fourier frequency transformations with the fMRI branch to capture global neural activity patterns. Third, we introduce a Multiscale Fusion Module that employs attention-based mechanisms to fuse complementary features from both modalities across multiple scales robustly. Finally, we integrate clinicopathological features via a dedicated fusion pathway, enriching the model with clinical context for precise biomarker prediction. Evaluated on the UCSF-PDGM dataset, our model achieved AUCs of 0.9779 and 0.9662 for IDH mutation and 1p/19q co-deletion prediction. Moreover, we achieved an AUC of 0.9739 across multiple datasets, confirming our method’s robustness. We employed SHapley Additive exPlanations to quantify feature contributions, revealing patient’s age significantly impacts IDH mutation prediction, while IDH mutation itself dominates 1p/19q co-deletion prediction. These findings underscore the superior predictive performance and adaptability of our approach, demonstrating multi-modal DL’s transformative potential for glioma molecular subtyping. 
<br />
  

Our previous work "Frequency-Domain Enhanced MRI Fusion with Multi-Head Attention for IDH Status Prediction" has been accept by TERNATIONAL CONFERENCE ON METAVERSE AND CURRENT TRENDS IN COMPUTING (ICMCTC-2025) 
https://tmrn.org/icmctc/
    </p>
    <br />
    <h3 align="lift"> Datasets:</h3>
 <p align="justify">  
    <br />
    UCSF-PDGM | The University of California San Francisco Preoperative Diffuse Glioma MRI (https://www.cancerimagingarchive.net/collection/ucsf-pdgm/)
    <br />
    UPENN-GBM | Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health System (https://www.cancerimagingarchive.net/collection/upenn-gbm/)
 </p>
</div>

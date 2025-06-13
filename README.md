# Anomaly Detection in the Music Genre of Primus
## Project Completed   
This research project has been completed. You can find the final report [here](https://github.com/DmytroRi/Primus-Anomaly-Detector/blob/main/When%20Metal%20Misbehaves%20Classifying%20Primus%20with%20Audio%20ML.pdf).   
If you're interested in the project roadmap, including motivation, discussions, experiments and results, feel free to explore this README.
## Overview
This project explores the use of machine learning to detect anomalies in the musical style of the band Primus. Inspired by their unconventional sound and unique approach to music, this research aims to analyze their genre and identify elements that make their music stand out from traditional classifications.
![Primus](https://i.redd.it/ebll5yxz7teb1.jpg)

### Why am I doing it?
As a **HUGE** fan of the prog-metal band TOOL from LA, one day I decided to check out Primus's music and ended up loving it. While I wouldn't describe myself as a die-hard fan of Primus, I find some of their songs incredible. I was also impressed by this part of their [Wikipedia page](https://en.wikipedia.org/wiki/Primus_%28band%29): 
>... The music of Primus has been described as "thrash-funk meets Don Knotts, Jr." and "the Freak Brothers set to music". The Daily Freeman described the band's style as "a blend of funk metal and experimental rock". The A.V. Club described the band's music as "absurdist funk-rock". Primus have also been described as "prog rock" or "prog metal". AllMusic places Primus within the first wave of alternative metal bands, saying that they fused heavy metal music with progressive rock. Entertainment Weekly classified the band's performance as "prog-rock self-indulgence"...   


For context, some of my favorite artists include Iron Maiden, TOOL, Bruce Dickinson, Pink Floyd, Swans, Radiohead, Red Hot Chili Peppers, Deftones, Porcupine Tree, Opeth, Deafheaven, Nine Inch Nails, Deep Purple, Metallica, Rush, Korn, Van Halen, Kraftwerk and others. You can check my [account on RYM](https://rateyourmusic.com/~HopelessExistentialist).  

Recently, I started learning the basics of machine learning and wanted to apply my new knowledge to a field I’m passionate about.


### Project Goals
1. By analyzing their discography, I aim to declare an anomaly in music genre classification or find the genre that best "suits" their style.
2. This project is not about creating a perfect model but about exploring, making mistakes, and improving along the way.

**Important Note**   
The classification of music genres is subjective! There are no strict rules that define clear borders between two pieces of music from different genres. However, we have some basic divisions created by fans and listeners, such as rock, metal, hip hop, rap, etc.   
For research purposes, I will use classifications from Wikipedia as well as my own experience as a music enthusiast. Over the past six years, I’ve recorded details of more than 300 albums in my personal Excel table — and I’m still adding to it. 

### Progress
#### 03.05.2025
The Purity value by segmentation strategy and methods as of 03.05.2025:

![Plot](/stats/20250503.png)    

#### 08.05.2025
I considered applying the k-nearest neighbors (KNN) algorithm, as the initial data is already labeled. In contrast, k-means performs better on unlabeled data because it can discover clusters or outliers.

#### 20.05.2025
Due to the complexity of implementing algorithms in C++, it would be more efficient to continue the work using Python and use existing solutions from different libraries. Additionally, given the large dataset because of the new framing strategy, the data should be stored in a local SQLite database for easier access and management.   

#### 26.05.2025
Precision by Number of Neighbors (k) as of 26.05.2025:   
![Plot](/stats/20250526.png)   
The highest precision value is 48.57%. The next steps are:
  - Compute the recall value
  - Compute the confusion matrix
  - Validate the algorithm with a more distinct dataset

#### 27.05.2025
The Confusion Matrix as for 27.05.2025:
![Plot](/stats/20250531cm.png)   
>Detailed classification report:
>
>                            precision    recall  f1-score   support   
>        alternative_metal       0.38      0.30      0.33    344077    
>              black_metal       0.52      0.69      0.59    455874    
>      classic_heavy_metal       0.43      0.50      0.46    398556    
>              death_metal       0.53      0.53      0.53    345131   
>                hard_rock       0.47      0.47      0.47    345601   
>                 nu_metal       0.43      0.33      0.37    297358      
>                   primus       0.62      0.51      0.56    319524   
>             thrash_metal       0.50      0.47      0.49    357040   
>   
>                 accuracy                           0.49   2863161   
>                macro avg       0.48      0.47      0.48   2863161   
>             weighted avg       0.48      0.49      0.48   2863161
>       

#### 28.05.2025
The Confusion Matrix as for 28.05.2025 (no Primus):
![Plot](/stats/20250531cm_noprimus.png)   
>Detailed classification report:
>
>                            precision    recall  f1-score   support
>       alternative_metal       0.41      0.32      0.36    344077
>             black_metal       0.54      0.70      0.61    455874
>     classic_heavy_metal       0.46      0.53      0.49    398556
>             death_metal       0.55      0.54      0.54    345131
>               hard_rock       0.51      0.52      0.52    345601
>                nu_metal       0.46      0.35      0.40    297358
>            thrash_metal       0.53      0.49      0.51    357040
>
>                accuracy                           0.50   2543637
>               macro avg       0.50      0.49      0.49   2543637
>            weighted avg       0.50      0.50      0.50   2543637
>

The next step would be to improve the nu-metal and alternative-metal datasets, as well as to adjust the duration of each subset.

#### 01.06.2025
By this time, I have tested the algorithm with a more diverse dataset containing 10 genres. The results were acceptable, with especially good performance on classical, metal, jazz, and pop music. Therefore, I can say that the algorithm works well, and the current main issue is likely feature engineering.   

Here is the confusion matrix for the external dataset:   
![Plot](/stats/20250529cm_extern.png)  
>Detailed classification report:   
>
>                      precision    recall  f1-score   support
>             blues       0.70      0.67      0.68     60180
>         classical       0.81      0.89      0.85     60208
>           country       0.60      0.59      0.59     60200
>             disco       0.49      0.41      0.45     60184
>            hiphop       0.63      0.46      0.53     60328
>              jazz       0.65      0.76      0.70     60218
>             metal       0.61      0.80      0.69     60153
>               pop       0.63      0.75      0.69     60140
>            reggae       0.68      0.51      0.59     60156
>              rock       0.49      0.46      0.48     60199
>
>          accuracy                           0.63    601966
>         macro avg       0.63      0.63      0.62    601966
>      weighted avg       0.63      0.63      0.62    601966
>
    
The styling of the other confusion matrices has also been updated. Each cell now shows not only the color but also the percentage relative to all frames from the genre.   
   
A few days ago, I also tried to visualize the dataset using Principal Component Analysis (PCA). Due to the large size of the dataset, the plot doesn't provide much detailed information, but some outliers (e.g. black metal) can be seen.   
Visualization of the project dataset:   
   
![Plot](/stats/20250531_datasetplot.png)  
    
Visualization of the external dataset:   
![Plot](/stats/20250530_datasetplot_extern.png)    
    
As already mentioned, the biggest challenge at this stage is feature engineering. After completing the binary classification on the project dataset, the feature list should be expanded with new parameters (e.g. additional MFCCs, Zero-Crossing Rate, Band Energy Ratio, etc.).   
I have already tried adding the first and second derivatives to the feature table of the external dataset (resulting in 39 features total), but this actually worsened the results:    
![Plot](/stats/20250602__deltasperformance.png)  


#### 04.06.2025
I have just got the following results of my binary classification task (Primus vs. Not Primus):
![Plot](/stats/20250604_binaryclassification.png)     
    
The highest precision was achieved with k = 10, reaching 91.46%.   
![Plot](/stats/20250604_binaryclassificationprecision.png)   

Based on these results, the following conclusions may be made:    
  1. The classifier is very good at identifying Non-Primus material.
  2. It struggles to recognize Primus correctly.

In other words, the model has very high specificity but a low sensitivity for Primus.
      
#### 05.06.2025
There are now some classification results based on my new datasets but first, a quick summary of the changes:
- The number of features was increased by 52, with only one feature set per song.
- New features include means and variances computed across: 20 MFCCs, sprectral centroid, spectral bandwidth, specral roll-off, root mean squaree energy, zero crossing rate and dynamic tempo.
- The framing strategy is now sample-based rather than duration-based.
- The sampling rate was manually reduced to 15 kHz.
- All feature values are z-scored before classification.

The reduced feature table size now allows for better dataset visualization:    
![Plot](/stats/20250605_datasetplot.png)    
    
While clear-cut clusters are still difficult to identify, we can observe some intuitive groupings — e.g. black metal and death metal are neighbors as are classic heavy metal and hard rock. Interestingly, Primus appears somewhat isolated, though with overlaps with other genres. This becomes even more apparent in a binary visualization:    
![Plot](/stats/20250605_datasetbinaryplot.png)    
     
Classification Results:   
- Multiclass classification precision: 59.26%
- Binary classification precision: 96.30%
- Best performance in both cases was achieved with k = 3   
![Plot](/stats/20250605_precision_v2.png)

Confusion Matrices for both classification tasks:    
![Plot](/stats/20250605cm_v2.png)    
     
![Plot](/stats/20250605_binaryclassification_v2.png)    

##### Interpretation of Results:
1. Binary experiment
    - High hit-rate — ~92% of Primus tracks are correctly identified.
    - Clean background — ~3 % of non-Primus tracks are mistakenly tagged as Primus.
    - F-score is ca. 93% — a solid, operational Primus detector.
  
2. Muliclass experiment
    - Recall is ca. 92% — the algorithm rarely misses Primus.
    - Frequent false positives are:   
           - Alt-metal (14%)   
           - Hard-rock (9%)   
   - Other genre predictions remain weak.
  
##### Practical conclusions:
1. Yes/No-Primus filter   
   Performs very well and is reliable for most use cases.
2. Multiclass precision   
   Still limited — likely due to either:
   - An insufficient number of features
   - Bad examples from overlapping genres

##### Next Steps:
1. Test on the external dataset used in previous experiments.
2. Rebuild the dataset using an equal number of songs per subgenre and compute mean/variance on a per-song basis.
3. Reframe songs into segments for feature aggregation — increasing the number of rows and targeting equal-sized genre-specific samples. 
    
#### 07.06.2025
This update presents the results from the steps described in the previous chapter. The precision on the external dataset increased from 63.15% to 67.50%. Below are the new confusion matrix and precision plot. Due to the smaller number of items in the new dataset, precision was calculated for k values in range from 1 to 31:   
Confusion Matrix:    
![Plot](/stats/20250605cm_externaldatasetv2.png)    
Precision plot:   
![Plot](/stats/20250605_precision_externaldatasetv2.png)    

##### Rebalanced Dataset: 60 Songs per Subgenre   
The dataset was then rebuilt to contain exactly 60 songs per subgenre. Below is a visualization of the updated dataset:   
![Plot](/stats/20250606_datasetplot_60songspergenre.png)    
As a result:
- Multiclass precision increased from 59.26% to 63.10%.    
- Binary classification precision slightly decreased from 96.30% to 95.24%.   
![Plot](/stats/20250605_precision_v2_60songspergenre.png)
    
Confusion Matrices:    
Multiclass    
![Plot](/stats/20250606cm_v2_60songspergenre.png)    
Binary    
![Plot](/stats/20250606_binaryclassification_v2_60songspergenre.png)    

##### Final Experiment: 3-Part Segmentation per Song     
As a closing experiment, each of the 60 songs per genre was divided into three equal-length parts, significantly increasing the dataset size.    
Results:
- Multiclass precision improved to 81.75%.   
- Binary precision dropped to 91.67%.
     
Confusion Matrices:   
Multiclass    
![Plot](/stats/20250606cm_v2_60songspergenre_3partdivision.png)    
Binary    
![Plot](/stats/20250606_binaryclassification_v2_60songspergenre_3partdivision.png)    
     
Dataset Visualizations (Three-Part Segmentation):    
![Plot](/stats/20250606_datasetplot_60songspergenre_3partdivision.png)    
![Plot](/stats/20250606_datasetplot_60songspergenre_3partdivision_binaryplot.png)    

# Anomaly Detection in the Music Genre of Primus
## Overview
This project explores the use of machine learning to detect anomalies in the musical style of the band Primus. Inspired by their unconventional sound and unique approach to music, this research aims to analyze their genre and identify elements that make their music stand out from traditional classifications.
![Primus](https://i.redd.it/ebll5yxz7teb1.jpg)

### Why am I doing it?
As a **HUGE** fan of the prog-metal band TOOL from LA, one day I decided to check out Primus's music and ended up loving it. While I wouldn't describe myself as a die-hard fan of Primus, I find some of their songs incredible. I was also impressed by this part of their [Wikipedia page](https://en.wikipedia.org/wiki/Primus_%28band%29): 
>... The music of Primus has been described as "thrash-funk meets Don Knotts, Jr." and "the Freak Brothers set to music". The Daily Freeman described the band's style as "a blend of funk metal and experimental rock". The A.V. Club described the band's music as "absurdist funk-rock". Primus have also been described as "prog rock" or "prog metal". AllMusic places Primus within the first wave of alternative metal bands, saying that they fused heavy metal music with progressive rock. Entertainment Weekly classified the band's performance as "prog-rock self-indulgence"...   


For context, some of my favorite artists include Iron Maiden, TOOL, Bruce Dickinson, Pink Floyd, Radiohead, Red Hot Chili Peppers, Deftones, Porcupine Tree, Nine Inch Nails, Deep Purple, Metallica, Rush, and others.   

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

The next step would be to improve the nu-metal and alternative-metal algorithms, as well as to adjust the duration of each subset.

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

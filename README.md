# Predict Student Performance (PSP) from Game Play
Predict Student Performance (PSP) from Game Play

Dataset link: [https://www.kaggle.com/competitions/predict-student-performance-from-game-play/](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/)

EDA:
1. Events plot for each level

Plan of implementation:
1. Observed the value counts of the various events
2. There are only 3 checkpoint events, only one at each of the three levels.
   1. Thus, aggregate the dataset to simplify the problem
   2. Following is the feature engineering plan:
      1. Get value counts for events at each level group
      2. Get the elapsed time at each checkpoint
      3. (OPTIONAL) Get the elapsed time for each level-group
   3. Target manipulation
      1. Get correctness for each of the 18 questions
3. Then we would get aggregate dataset 
   1. each data instance would represent
      1. session_id
      2. aggregate event values
      3. question correctness
4. Modelling idea
   1. Take the aggregate values
   2. Predict the correctness of 18 questions
5. Type of problem: Multi-label Classification (MLC)
   1. [MLC MLM Reference](https://machinelearningmastery.com/multi-label-classification-with-deep-learning/)
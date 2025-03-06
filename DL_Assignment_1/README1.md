The Dataset is collected from binary classification of insurance cross selling, While working with the dataset, found that it's highly imbalanced. Therefore ADASYN was implemented to synthetically represent minority class in smaller sample set. 
Before implementing adasyn the model was resulting 85% accuracy, but it was unable to generalize and accurately predict the minority class.
After implementing ADASYN, the accuracy dropped to 68-70%, using multiple hyperparameter tuning, even though the overall accuracy dropped, the model's ability to predict the minority class has improved.
however, There is still room for improvement to optimize the model. trying more hidden layer combination and using other activation function such as sigmoid, tanh etc could improve the model's performance.
To, conclude, follwing the standard procedure to train a model using pytorch, a decent model has been trained, but its hasn't been able to predict optimum results yet.

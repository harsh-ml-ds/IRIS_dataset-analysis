# IRIS_dataset-analysis
Creating a machine learning model to analyse the Iris dataset
1. Read the dataset with changing column-names to =['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'target']
2. Store the values of ['sepal-length', 'sepal-width', 'petal-length', 'petal-width'] to x and ['target'] to y
3. Scale x using StandardScaler()
4. Usina PCA fit x
5. Make a Dataframe using 2 Principal Components and check which one is contributing more by using "pca.explained_variance_ratio_ ".
6. Using concatination, add the left-over column ['target'] to the above DataFrame
7. Using Scatter plot, check for the observations ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'] in ['target'] and also include check for PC1 and PC2
8. change the column names again
9. Take values of [sepal-length, sepal-width, petal-length, petal-width] in x and [Class] in y
10. use train_tets_split
11. use StandardScaler on x_train and x_test
12. use LDA on x_train and x_test
13. scatter plot(x_test, np.zeros(len(x_test)), c=y_test)
14. Use ML model RandomForesClassifier on (x_train, y_train) and predict y from x_tets
15. Calculate accuracy of model

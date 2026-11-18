# Experiment 6 – Principal Component Analysis (PCA) on Iris Dataset

## AIM
To reduce the 4-dimensional Iris dataset into 2 principal components using Principal Component Analysis (PCA) and visualize the separation of different Iris species.

## ALGORITHM
1. Import the required Python libraries.
2. Load the Iris dataset using load_iris().
3. Separate the input features (X) and target labels (y).
4. Apply PCA with n_components=2 to reduce the dataset from 4 dimensions to 2 dimensions.
5. Use fit_transform() to fit the PCA model and transform the data.
6. Display the explained variance ratio of the two principal components.
7. Visualize the transformed data using a scatter plot.
8. Differentiate the Iris species using different colors.

## PROGRAM

python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA
pca = PCA(n_components=2)
Z = pca.fit_transform(X)

# Display explained variance ratio
print('Explained variance ratio:',
      np.round(pca.explained_variance_ratio_, 4))

# Visualize PCA results
plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Iris')
plt.show()


## DETAILED EXPLANATION

1. *Load Iris Dataset:* The Iris dataset is loaded using load_iris() from the Scikit-learn library. It contains 150 samples, 4 features, and 3 species.

2. *Apply PCA:* PCA(n_components=2) creates two principal components from the original four features.

3. *Fit and Transform:* fit_transform(X) calculates the principal components and projects the original data into a 2-dimensional space.

4. *Explained Variance Ratio:* explained_variance_ratio_ displays the proportion of total variance captured by each principal component.

5. *Visualization:* A scatter plot is created using Matplotlib. Different colors represent the three Iris species.

## SAMPLE OUTPUT

*Explained Variance Ratio:*

text
[0.9246 0.0531]


*Visualization:*

The scatter plot displays the Iris dataset using two principal components (PC1 and PC2). Different colors represent the three Iris species.

## RESULT

Thus, the 4-dimensional Iris dataset was successfully reduced to 2 principal components using PCA. PC1 captures approximately 92.46% of the total variance, while PC2 captures approximately 5.31%.

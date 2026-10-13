# Iris Flower Classification Using Support Vector Machines

A straightforward machine learning project that uses Linear and Non-Linear Support Vector Machines (SVM) to classify Iris flowers. I used Principal Component Analysis (PCA) to shrink the original four features down to two components so I could actually plot and visualize the decision boundaries.

## What's Inside
- Building and training a **Linear SVM** model.
- Building and training a **Non-Linear SVM** model with an RBF kernel.
- Using **PCA** for 2D visualization.
- Generating confusion matrices, accuracy scores, and classification reports.
- Comparing how the linear vs. non-linear boundaries handle the data.

## The Dataset
Loaded straight from `scikit-learn`:
- **150 total samples** across 3 species (*Setosa*, *Versicolor*, *Virginica*)
- **4 features:** Sepal length, sepal width, petal length, petal width
- **Split:** 80% training / 20% testing

## Tech Stack
- Python
- Scikit-learn
- Pandas / NumPy
- Matplotlib
- Google Colab

## Step-by-Step Workflow
1. **Load data:** Import the Iris dataset via `sklearn.datasets`.
2. **Train/Test Split:** Split the data 80/20.
3. **Scale:** Standardize features using `StandardScaler` (crucial before running PCA).
4. **PCA:** Reduce 4 dimensions to 2 principal components.
5. **Train Models:** 
   - Linear SVM (`kernel='linear'`, `C=1.0`)
   - Non-Linear SVM (`kernel='rbf'`, `C=1.0`)
6. **Evaluate:** Compare accuracy, precision, recall, and look at the confusion matrices.
7. **Plot:** Visualize the decision regions, support vectors, and a final accuracy comparison chart.

## Results

| Model | Kernel | Accuracy |
| :--- | :--- | :--- |
| Linear SVM | Linear | 0.9 |
| Non-Linear SVM | RBF | 0.9 |

## Running the Code
1. Open up `SVM_Iris_Classification.ipynb` in Google Colab or Jupyter Notebook.
2. Make sure you have the dependencies installed:
   ```bash
   pip install numpy pandas matplotlib scikit-learn

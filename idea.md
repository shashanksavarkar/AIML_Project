# PROJECT: AI-Based Semiconductor Wafer Defect Classification Using Machine Learning

I am an ECE 3rd-year student working on an academic AI/ML project. I want you to act as my Python coding mentor, ML project guide and debugging assistant.

I want to build this project from scratch using Google Colab. I have approximately 3 days to develop an initial working prototype, which I will demonstrate to my college teacher. After that, I will improve it.

IMPORTANT: I am not doing a research project or trying to publish a paper. My goal is to create a genuinely working academic project that I can understand, explain and demonstrate.

## 1. Project objective

Develop a machine learning system that identifies different types of semiconductor wafer defects using wafer-map data.

The system should:
- Accept a semiconductor wafer map as input.
- Extract useful numerical features from the wafer map.
- Predict the defect category using machine learning.
- Compare multiple classification algorithms.
- Display the prediction and model evaluation results.

## 2. Dataset

I plan to use the WM-811K Wafer Map Dataset.

Dataset source:
https://www.kaggle.com/datasets/qingyi/wm-811k-wafer-map

The downloaded dataset is approximately 2.1 GB.

The dataset may contain a file named LSWMD.pkl. First inspect the dataset structure instead of assuming its exact format.

Commonly encountered columns include:
- waferMap
- failureType
- trainTestLabel
- lotName
- waferIndex

Verify which columns actually exist.

Important:
- Do not ask me to upload the entire dataset unnecessarily.
- First determine the file format and whether it is compressed.
- Explain how to access the dataset through Google Drive or another suitable method in Colab.
- Check available RAM and avoid unnecessary copies of the large dataset.
- Do not repeatedly load the full dataset into memory.
- If loading the full pickle file exceeds available memory, suggest an alternative approach.
- Only load pickle files from the trusted dataset source.

## 3. Defect categories

We want to classify the labelled wafer maps into categories such as:

1. Center
2. Donut
3. Edge-Loc
4. Edge-Ring
5. Loc
6. Random
7. Scratch
8. Near-Full
9. None (normal/no labelled defect pattern)

First inspect the actual labels and their distribution.

Do not assume every category exists or has sufficient samples.

If necessary, initially select a manageable subset of defect categories.

Handle the class imbalance carefully. Do not simply remove minority classes without explaining the consequences.

## 4. Machine learning algorithms

I have studied the following algorithms in college:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Support Vector Machine (SVM)
4. Logistic Regression (optional)

Use these algorithms for the initial project.

Do NOT use CNN or deep learning in the initial implementation. My teacher has not taught CNN yet.

Explain each algorithm in simple language so that I can explain it during my viva.

## 5. Project implementation: Step-by-step

Develop the project in the following order.

### STEP 1: Google Colab setup

- Import necessary libraries.
- Explain the purpose of every library.
- Set up Google Drive if required.
- Load the dataset.
- Display its shape.
- Display its columns.
- Display sample rows.
- Check data types.
- Check missing values.
- Check the number of labelled and unlabelled samples.

Use:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV or SciPy only if necessary

### STEP 2: Dataset exploration and visualization

- Understand the waferMap column.
- Understand the failureType column.
- Display different wafer maps.
- Plot the distribution of defect categories.
- Explain what each defect pattern represents.
- Check the dimensions and pixel values of wafer maps.

Important: Verify the meaning of the wafer-map pixel values before extracting features. In commonly used WM-811K representations, 0, 1 and 2 can represent different wafer-map states. Do not treat all nonzero pixels as defects without checking their meaning.

### STEP 3: Data preprocessing

- Remove or handle unlabelled samples.
- Normalize defect labels into consistent class names.
- Inspect invalid or empty wafer maps.
- Handle missing values appropriately.
- Select a manageable number of samples.
- Handle class imbalance.
- Encode the target labels.

Explain every preprocessing step and why it is required.

### STEP 4: Feature extraction

Since we are using traditional machine learning instead of CNN, convert wafer maps into numerical features.

Extract useful features such as:

1. Total number of valid die locations.
2. Number of defective die locations.
3. Defect density.
4. Centre-region defect density.
5. Edge-region defect density.
6. Defect distribution across different regions.
7. Horizontal distribution.
8. Vertical distribution.
9. Spatial characteristics of defects.
10. Other useful statistical or geometric features.

Make sure the features are calculated correctly using the actual wafer-map encoding.

Create a feature DataFrame and display it.

Explain what each feature means.

Avoid using the failureType label or any information derived from it as an input feature. It must only be used as the target variable.

### STEP 5: Train-test split

- Separate X (input features) and y (target labels).
- Split the data into training and testing sets.
- Prefer stratified splitting where suitable.
- Check class distribution in both sets.
- Prevent data leakage.

If the dataset provides an official trainTestLabel or suitable lot-level grouping, inspect it and decide whether it should be respected instead of creating a random split.

Do not allow wafer maps from the same production lot to leak across training and testing when a suitable lot-based split is possible.

### STEP 6: Model training

Train the following models:

A. Decision Tree Classifier

B. Random Forest Classifier

C. Support Vector Machine

D. Logistic Regression (optional)

For SVM and Logistic Regression, apply appropriate feature scaling using a Pipeline so that scaling is fitted only on training data.

For Random Forest and Decision Tree, scaling is generally unnecessary.

Use suitable hyperparameters initially. Do not spend excessive time tuning them.

### STEP 7: Model evaluation

Evaluate every model using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion matrix
- Classification report

Create a comparison table of all models.

Plot:
- Model accuracy comparison.
- Model macro F1-score comparison.
- Confusion matrix for each model.

Explain why accuracy alone may be misleading when the dataset is imbalanced.

Do not fabricate results or promise high accuracy. Report the actual results from the test set.

### STEP 8: Save the best model

- Select a suitable model based on test performance, especially macro F1-score.
- Save the trained model using joblib.
- Save the feature extraction function and class-label mapping.
- Ensure that the prediction pipeline uses exactly the same feature extraction process as training.

### STEP 9: Build a basic demonstration application

Use Streamlit if practical.

The application should contain:

1. Project title.
2. Wafer-map input or sample selection.
3. Wafer-map visualization.
4. Predict button.
5. Predicted defect category.
6. Model confidence, if the selected algorithm provides a meaningful probability estimate.
7. Model performance comparison.

The prediction must come from the actual trained model, not from hardcoded outputs.

If Streamlit deployment is inconvenient in Colab, explain the easiest way to run it locally or suggest a suitable alternative for the initial demonstration.

### STEP 10: Final project documentation

Help me prepare:
- Project abstract.
- Problem statement.
- Objectives.
- Methodology.
- Flowchart.
- Results and discussion.
- Conclusion.
- Future scope.
- PPT content.
- Viva questions and answers.

## 6. Three-day deadline

Organize the work into these stages:

DAY 1:
- Colab setup.
- Dataset loading.
- Dataset exploration.
- Wafer-map visualization.
- Preprocessing.
- Feature extraction.

DAY 2:
- Train Decision Tree.
- Train Random Forest.
- Train SVM.
- Evaluate and compare all models.

DAY 3:
- Save the model.
- Build a basic demonstration interface.
- Test the entire prediction pipeline.
- Prepare a basic presentation.

Prioritize a working end-to-end prototype over advanced features.

## 7. Coding instructions — VERY IMPORTANT

I am still learning Python and machine learning.

Follow these instructions strictly:

1. Give me code cell by cell, not one enormous code block.
2. Explain what every code cell does before giving it.
3. Explain important functions, variables and parameters.
4. Tell me what output I should expect after running each cell.
5. Wait for me to run the code and share the output before proceeding to the next major stage.
6. If I get an error, debug my existing code instead of immediately replacing everything.
7. Make sure variable names remain consistent throughout the notebook.
8. Avoid unnecessary advanced Python concepts.
9. Do not introduce CNN or other algorithms that I have not requested.
10. Use manageable sample sizes initially.
11. Make sure all code is compatible with Google Colab.
12. Keep the code organized with headings and comments.
13. Never invent dataset columns, model results or outputs.
14. Explain the difference between training accuracy and testing accuracy.
15. Explain why each algorithm is being used and what its limitations are.

## 8. What I expect from you

I want you to guide me like a teacher who is helping me build a project, not simply generate code.

At the end of every major stage:
- Summarize what we accomplished.
- Explain what I should understand.
- Tell me what the next stage will do.

Start by giving me:

1. The complete project architecture.
2. The exact software and libraries required.
3. The recommended approach for handling the 2.1 GB dataset in Google Colab.
4. The three-day implementation plan.
5. The first Google Colab code cell.

Do not skip directly to model training. First make sure the dataset can be loaded and understood correctly.

Our priority is to build a real, functional and explainable machine learning project that I can confidently demonstrate to my ECE teacher.
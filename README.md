# VEBOCH-SOCIAL-MEDIA-ADDECTION
Social Media Use &amp; Mental Well-being: Linear Regression  Predict a person's Mental Well-being Score from their social-media habits and related self-reported scores, using a scikit-learn preprocessing pipeline and Linear Regression.  
Social Media Use & Mental Well-being: Linear Regression

Predict a person's Mental Well-being Score from their social-media habits and related self-reported scores, using a scikit-learn preprocessing pipeline and Linear Regression.

Task: supervised regression
Target (y): Mental_Wellbeing_Score
Model: Linear Regression, with automatic feature selection (Lasso) inside a Pipeline
Main file: social_media_addection_mental_wellbeing.ipynb
Table of contents
Dataset
Repository structure
Environment setup
How to run
Methodology
Evaluation
Visualizations
Decisions log
Limitations
Troubleshooting
1. Dataset

File: social_media_addiction_mental_wellbeing.csv Size: 1,500 rows × 30 columns, one row per person.

Group	Columns
Identifier (dropped)	User_ID
Demographics	Age, Gender, Occupation, Relationship_Status
Usage behaviour	Primary_Platform, Daily_Usage_Hours, Platforms_Used_Count, Posts_Per_Week, Notifications_Per_Day, Late_Night_Usage, First_Check_Morning, Screen_Free_Time_Hrs
Psychological scores	FOMO_Score, Social_Comparison_Score, Validation_Seeking_Score, Scroll_Without_Purpose, Anxiety_Score, Depression_Score, Loneliness_Score, Self_Esteem_Score
Attempts to cut back	Tried_To_Cut_Back, Failed_To_Cut_Back
Sleep / lifestyle	Sleep_Hours, Sleep_Quality_Score, Productivity_Loss_Score, Offline_Relationship_Quality, Physical_Activity_Hrs_Week
Target	Mental_Wellbeing_Score (continuous, roughly 1.5 to 10)
Excluded (leakage)	Addiction_Level (Low / Moderate / High / Severe)

Data quality

No duplicate rows.
About 3 to 5% missing values in most feature columns. The target and First_Check_Morning have none.
Column types: numeric scores, Yes/No flags, one ordered text column (First_Check_Morning), and unordered text columns.
2. Repository structure
.
├── social_media_addection_mental_wellbeing.ipynb   # main notebook (preprocessing, model, plots)
├── social_media_addiction_mental_wellbeing.csv      # dataset
├── requirements.txt                                 # Python dependencies
└── README.md                                        # this file
3. Environment setup
What is an environment, and why use one?

An environment is an isolated folder that holds its own copy of Python and its own installed libraries. Different projects often need different library versions; an environment keeps them from clashing, and lets anyone recreate your setup exactly. The project's dependencies are listed in requirements.txt.

Libraries used
Library	Used for
pandas	loading and manipulating the table
numpy	numeric operations
scikit-learn	train/test split, imputation, scaling, encoding, ColumnTransformer, Pipeline, Lasso feature selection, Linear Regression, metrics
matplotlib	plotting
seaborn	statistical plots (histogram, boxplot, heatmap)
jupyter / ipykernel	running the notebook
Option A: Google Colab (the notebook was written for this)

No installation needed: Colab already includes all the libraries above.

Open the notebook in Colab (File > Upload notebook).
Upload the CSV to the Colab file panel at /content/sample_data/ (the notebook reads /content/sample_data/social_media_addiction_mental_wellbeing.csv).
Run the cells in order (see How to run).

Colab sessions reset, so re-upload the CSV and re-run every cell from the top after a restart.

Option B: Run locally with venv

Requires Python 3.9 or newer.

bash
# 1) create the environment (once)
python -m venv .venv

# 2) activate it (every time you open a new terminal)
#    Windows:
.venv\Scripts\activate
#    macOS / Linux:
source .venv/bin/activate

# 3) install the dependencies
pip install -r requirements.txt

# 4) start Jupyter
jupyter notebook

Then open the notebook and change the data path in the loading cell to a relative one:

python
data = pd.read_csv("social_media_addiction_mental_wellbeing.csv")
Option C: Run locally with Conda
bash
conda create -n wellbeing python=3.11
conda activate wellbeing
pip install -r requirements.txt
jupyter notebook
Check what is installed
bash
pip list                      # all installed packages with versions
python --version

or inside a notebook cell:

python
import sys, pandas, numpy, sklearn, matplotlib, seaborn
print(sys.version)
for lib in (pandas, numpy, sklearn, matplotlib, seaborn):
    print(lib.__name__, lib.__version__)
Tested versions

The pipeline was re-run end to end with:

Package	Version
Python	3.12.3
pandas	3.0.2
numpy	2.4.4
scikit-learn	1.8.0
matplotlib	3.10.8
seaborn	0.13.2

The lower bounds in requirements.txt are conservative; the code was only executed with the versions above. To freeze your exact versions for reproducibility, run pip freeze > requirements-lock.txt.

4. How to run

Run the notebook top to bottom, in this order:

Imports and data loading. Make sure LassoCV is imported (from sklearn.linear_model import LassoCV).
Exploration: shape, info, describe, nunique, isnull().sum().
Define x and y, then the train/test split.
Column groups, then the preprocessor and model (ColumnTransformer + Pipeline), and model.fit(...).
Evaluation (see section 6). Run it right after the model is fitted.
Visualization section. Note that this section reloads the data and redefines df, x, y, x_train, x_test, so run the model and evaluation cells before it.
5. Methodology
5.1 Columns removed from the features
Column(s)	Why
User_ID	identifier, carries no information
Addiction_Level	data leakage: correlation of about -0.89 with the target. Keeping it makes results look better (R² near 0.83) without the model learning anything real
Age, Posts_Per_Week, Sleep_Hours, Depression_Score	near-zero correlation with the target; removing them did not change accuracy
5.2 Train / test split

80% training (1,200 rows) and 20% testing (300 rows), random_state=42 for reproducibility.

5.3 Preprocessing

All preprocessing is fit on the training set only and then applied to the test set, to avoid leakage. It is wrapped in a ColumnTransformer:

Column group	Columns	Steps
Numerical (15)	Daily_Usage_Hours, FOMO_Score, Anxiety_Score, ...	missing → median, then StandardScaler
Ordinal + Yes/No (4)	First_Check_Morning, Late_Night_Usage, Tried_To_Cut_Back, Failed_To_Cut_Back	missing → most frequent, then OrdinalEncoder with an explicit order
Categorical (4)	Gender, Occupation, Relationship_Status, Primary_Platform	missing → most frequent, then OneHotEncoder

Explicit orders used by the OrdinalEncoder:

First_Check_Morning: After 30 min (0) < 5-30 min (1) < Within 5 min (2)
Yes/No columns: No (0) < Yes (1)

An explicit order matters: without it the encoder sorts alphabetically, which would place 5-30 min below After 30 min.

5.4 Model
ColumnTransformer (preprocessing)
        ↓
SelectFromModel(LassoCV)     ← automatic feature selection
        ↓
LinearRegression

Everything is inside a single Pipeline, so the imputation, scaling, encoding and feature selection are all learned from the training data only. With 35 columns after encoding, Lasso kept 28 and dropped 7.

6. Evaluation

Add this cell right after model.fit(x_train, y_train):

python
from sklearn.metrics import mean_absolute_error

predictions = model.predict(x_test)
print("R2  :", round(r2_score(y_test, predictions), 3))
print("MAE :", round(mean_absolute_error(y_test, predictions), 3))
print("RMSE:", round(mean_squared_error(y_test, predictions) ** 0.5, 3))

Reference results (300 test rows, random_state=42):

Metric	Value	Meaning
R²	0.793	the model explains about 79% of the variance in the well-being score
MAE	0.483	the average prediction error is about half a point
RMSE	0.612	typical error, penalising large mistakes more

R² from a single split varies a little with random_state. A 5-fold cross-validated R² on the same data is about 0.77, which is the more stable number to report:

python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5, scoring="r2")
print(scores.mean().round(3))

Run this on the feature set x defined before the visualization section.

7. Visualizations
Plot	What it shows
Histogram + KDE of the target	distribution of Mental_Wellbeing_Score (roughly bell-shaped, mean about 6.5)
Horizontal bar chart of correlations	correlation of each numeric column with the target

Reading the correlation chart: a negative bar means that as the column increases, well-being tends to decrease. The strongest negative relationships are with the "compulsive use" indicators (Scroll_Without_Purpose, FOMO_Score, Social_Comparison_Score, Daily_Usage_Hours); positive ones include Sleep_Quality_Score and Self_Esteem_Score.

8. Decisions log: what was tried and why
Tried	Outcome
QuantileTransformer instead of StandardScaler	Rejected. Cross-validated R² dropped from about 0.77 to about 0.70. Most columns are not skewed, and the transform distorts their roughly linear relation with the target
SelectKBest(k=15)	Rejected. R² dropped to about 0.71 because it scores each column alone and ignores combined effects
Lasso / RFECV feature selection	No gain in accuracy (about 0.77 either way), but gives a simpler model and a defensible, automatic selection
OrdinalEncoder on unordered text columns	Avoided. It invents a fake order (e.g. TikTok "greater than" Instagram); OneHotEncoder is used instead
Keeping Addiction_Level	Rejected. Leakage
Random Forest / Gradient Boosting	Worse than the linear model in testing (about 0.49 and 0.61 R²), suggesting the relationships are mostly linear
9. Limitations
Correlation is not causation. The model describes associations in this dataset; it does not show that any habit causes lower well-being.
The features are almost uncorrelated with each other (the strongest pair is about |r| = 0.08), which is unusual for real survey data and suggests the dataset may be synthetic or simulated. Conclusions may not transfer to real populations.
No hyper-parameter tuning and a single train/test split (cross-validation is suggested above).
Coefficients of scaled and unscaled columns are not directly comparable in size.
10. Troubleshooting
Problem	Cause and fix
NameError: name 'sns' is not defined (or plt, LassoCV, x_train)	The cell that defines it was not run. Run the cells from the top in order; for LassoCV, add it to the imports
FileNotFoundError when loading the CSV	The path is wrong. On Colab upload the file to /content/sample_data/; locally use a relative path
KeyError: ('a', 'b')	Selecting two columns needs double brackets: df[['a', 'b']]
ValueError: Input X contains NaN	A missing value was not imputed. Keep the imputers inside the Pipeline
categorical comes out empty	On pandas 3.x text columns have dtype str, not object. Use select_dtypes(exclude='number') instead of include=['object']
ModuleNotFoundError	Install the package: pip install seaborn (or whichever is missing), then restart the kernel
Results differ slightly between runs	Keep random_state=42 in train_test_split and the models
Author

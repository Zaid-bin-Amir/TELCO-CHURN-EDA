# Telco Customer Churn: Statistical Profiling, EDA and Naive Baseline

Prepared by: Zaid

## 1. Overview

This project is a structured exploratory data analysis (EDA) of the Telco Customer Churn dataset, followed by a deliberately simple majority-class baseline. The goal is not to build a good churn model. The goal is to understand the data honestly before any model exists, and to set a reference score that every later model must beat.

The work answers six questions in order. What does the data look like, and is it clean? How are the numerical columns distributed, and where do averages mislead? How imbalanced is the churn target, and what does that mean for metric choice? Which customer features are visibly tied to churn? How do the numerical features relate to each other and to churn? And what does a model that learns nothing actually score?

Everything in this repository was produced by code that runs from top to bottom without errors. The notebook contains its own saved outputs, so a reader can see every number and chart without running anything.

## 2. Files in this repository

| File | Purpose |
|---|---|
| `eda_churn.ipynb` | The main notebook: data checks, statistics, charts, correlation heatmap and baseline, with outputs saved |
| `EDA_FINDINGS.md` | A written summary of the three main feature-versus-churn insights, plus statistics and correlation notes |
| `BASELINE.md` | The majority-class baseline metrics and a plain explanation of why accuracy alone misleads |
| `README.md` | This report |

The notebook runs in Google Colab. On the first cell it asks for the CSV file if the file is not already present, and the rest runs without any edits.

## 3. The dataset

The data is the IBM sample Telco Customer Churn dataset, available on Kaggle. It has 7,043 customers and 21 columns. Each row is one customer, described by demographic fields such as gender, senior-citizen status, partner and dependents, by the services they hold such as phone, internet, online security, tech support and streaming, by account fields such as contract type, paperless billing and payment method, and by three numerical fields: tenure in months, monthly charges and total charges. The target column, `Churn`, says whether the customer left (Yes or No), which makes this a binary classification problem.

There are no duplicate customer IDs, so each row is a distinct customer.

## 4. Data quality: the TotalCharges quirk

The first thing the notebook does is check why `TotalCharges` loads as text instead of a number. The column contains 11 blank strings, and a careless reader might treat them as random missing values and impute the column average. The notebook tests that assumption instead of making it.

Every one of the 11 blank rows has a tenure of zero, and every customer with a tenure of zero has a blank. The two groups are identical. These customers had just signed up and had not been billed yet, so the blank means "nothing charged so far", which is zero. The correct fix is to convert the column to numbers and set those 11 values to 0, not to fill them with a typical value. Filling them with the mean would invent roughly 2,280 in charges for customers who had paid nothing.

The notebook also notes that several service columns, such as `OnlineSecurity` and `TechSupport`, contain a "No internet service" level, and `MultipleLines` contains "No phone service". These are genuine answers (the customer does not have that product), not missing data, and they should be kept as their own category.

## 5. Summary statistics and skewed distributions

The notebook computes the mean, median, standard deviation, variance, skewness and kurtosis for the numerical columns. Skewness is used because it quantifies a distribution's asymmetry, and the gap between mean and median is a quick way to see whether an average can be trusted.

Two distributions are clearly skewed.

The first is `TotalCharges`, with a skewness of 0.96. Its mean is about 2,280 while its median is about 1,395, a gap of roughly 885. A long tail of customers with many months of service and high bills pulls the mean upward, so the mean overstates what a typical customer has paid. The median is the safer summary. A log transform removes the right skew, though it overshoots to about -0.82, so it is a partial fix and not a perfect one. The implication for later work is that linear or distance-based models should receive a transformed or scaled version of this column.

The second is `SeniorCitizen`, with a skewness of 1.83. This column is a 0/1 flag, so the skew only reflects the fact that 16.2% of customers are senior citizens. Its mean is simply that share and its median is zero. The practical lesson is that any conclusion about seniors rests on a much smaller group than a conclusion about non-seniors, so it carries more uncertainty.

The notebook also warns about two columns where the skewness number is small but the mean is still misleading. `Tenure` has a skewness of only 0.24, yet its histogram is U-shaped: 21.0% of customers have six months or less and 14.7% have 66 months or more, while the middle is thin. Its mean of about 32 describes almost nobody. `MonthlyCharges` has a skewness of -0.22, yet it is bimodal, with about 1,400 customers paying roughly 18 to 25 per month and a second broad group paying 70 to 100. A single skewness figure hides both patterns, which is why the notebook shows the histograms and box plots next to the numbers.

## 6. The target and class imbalance

Of the 7,043 customers, 5,174 did not churn (73.46%) and 1,869 did (26.54%). The ratio is about 2.77 to 1. This is a moderate imbalance, and it matches the 26 to 27% positive rate that is typical for this dataset.

The consequence for metric choice is the most important idea in the project. When one class makes up nearly three quarters of the data, accuracy becomes a weak measure, because a model can score well just by favoring the majority class. Later models should therefore be judged on precision, recall and F1 for the churn class, and probably on PR-AUC or ROC-AUC, using stratified train and test splits so the class ratio is preserved in both parts.

## 7. Bivariate analysis: which features go with churn

The notebook examines four relationships between a feature and churn. Each has a table, a chart and a written interpretation.

### 7.1 Contract type

Churn depends heavily on the contract. Month-to-month customers churn at 42.71%, one-year customers at 11.27% and two-year customers at 2.83%. Month-to-month customers churn about fifteen times as often as two-year customers. They also make up 55% of all customers, so this single group accounts for 1,655 of the 1,869 churners, about 89% of all churn. The pattern is steady: the longer the commitment, the lower the churn.

The interpretation is careful on purpose. The data shows an association, not a cause. Customers who already plan to stay are likely the ones who pick a long contract, so the contract partly reflects loyalty that already existed and may not be creating it.

### 7.2 Tenure

Churners have a mean tenure of about 18 months and a median of only 10, while customers who stayed have a mean of about 38 and a median of 38. Grouping tenure into bands shows churn falling steadily: 52.94% in the first six months, then 35.89%, 28.71%, 20.39% and 14.42% in later bands, and only 6.61% for customers with more than five years of service.

The first year is clearly the danger period. There is also a statistical trap here called survivorship. The long-tenure group can only contain people who did not leave earlier, so part of the low churn at the end is a selection effect and not purely evidence that loyalty grows with time.

### 7.3 Monthly charges

Churners pay more on average, with a mean of 74.44 against 61.27 for customers who stayed. The relationship is not a straight line. The cheapest band (30 or less) churns at 9.80%, the 70 to 90 band churns at 37.80%, and the 50 to 70 band dips to 20.76% in between.

Looking one level deeper explains why. Of customers paying 30 or less, 92% have no internet service, and customers without internet churn at only 7.40%. Fiber optic customers pay about 91.5 per month on average and churn at 41.89%. So price is tangled up with service type, and the chart cannot show that price alone makes people leave. This is a good example of a confounded relationship.

### 7.4 Internet service and payment method

Two extra views confirm where churn concentrates. Customers paying by electronic check churn at 45.29%, compared with 15.24% to 19.11% for the other three payment methods. Fiber optic customers churn at 41.89%, compared with 18.96% for DSL. Both are strong signals worth carrying forward, with the reminder that they overlap with contract type and tenure.

## 8. Correlation analysis

The correlation heatmap covers tenure, monthly charges, total charges, senior-citizen status and churn (coded as 0 and 1). The notebook ranks every pair by strength and flags the notable ones.

The strongest relationship is between tenure and total charges (+0.83), followed by monthly charges and total charges (+0.65). This is not surprising once it is checked, because total charges is almost a derived column. Its correlation with tenure multiplied by monthly charges is 0.9996, and 94.4% of customers fall within 10% of that product. It is effectively a near-duplicate feature, which matters for feature engineering and for multicollinearity in linear models.

The most interesting finding is a sign flip. Monthly charges correlates positively with churn (+0.19) while total charges correlates negatively (-0.20). Both signs come from tenure: a high monthly bill goes with more churn, but a high lifetime total simply means the customer has stayed for a long time.

The weakest relationship is between tenure and senior-citizen status (+0.02), which are essentially unrelated. Senior status has only a modest link to churn (+0.15), and tenure has the strongest link to churn (-0.35), which is moderate at best.

The notebook also states the limits of the method. Pearson correlation only measures straight-line association, so it understates the uneven monthly-charges pattern seen earlier. Correlations with a 0/1 target are compressed. And no correlation, however strong, proves that one variable causes another.

## 9. The naive baseline

The baseline is a scikit-learn `DummyClassifier` with the most-frequent strategy. It ignores every feature and always predicts "No Churn". Because this task is EDA, it is fitted and scored on the full dataset, and the churn class is treated as the positive class.

Its results are an accuracy of 0.7346, and a precision, recall and F1 for the churn class of exactly 0.00. It catches none of the 1,869 customers who left. The confusion matrix has 5,174 correct "No" predictions, 1,869 missed churners, and nothing else.

The core lesson is that a model which learns nothing and finds nobody still scores 73.46% accuracy. If someone only reported accuracy, this useless model would look decent. The churn-class metrics expose it immediately. For contrast, the notebook also scores a classifier that always predicts "Churn". It gets only 26.54% accuracy, yet its recall is 1.00 and its F1 is 0.42, which is higher than the F1 of the "always No" model. The ranking of the two flips depending on the metric, so the metric must match the business goal of finding customers who are about to leave.

The baseline is also a meaningful reference point and not a joke. Any real model must beat 73.46% accuracy, and more importantly must achieve a churn-class F1 well above zero and above the 0.42 of blind guessing. A later model that reports about 75% accuracy is only around 1.5 points better than doing nothing, so its churn-class precision and recall must be checked before it is called good.

## 10. What a reader will learn from this project

A reviewer or student who studies this repository can take away the following lessons.

**Test assumptions about missing data.** The blank `TotalCharges` values look like random missingness but turn out to be a structural zero tied to new customers. Checking the relationship with tenure leads to the correct fix, while a default imputation would have quietly introduced wrong values.

**Do not trust a single summary number.** The mean and skewness tell only part of the story. The gap between mean and median reveals skew in `TotalCharges`, while `tenure` and `MonthlyCharges` show that a distribution can be U-shaped or two-humped even when its skewness looks harmless. Always plot the distribution beside the statistic.

**Recognize class imbalance before modeling.** A 73/27 split changes which metrics are meaningful. Stating the imbalance and its consequences up front prevents a headline accuracy figure from fooling anyone later.

**Interpret relationships, do not just display them.** Each chart in the notebook comes with a paragraph explaining what it shows and what it does not. The contract result is tempered by self-selection, the tenure result by survivorship, and the monthly-charges result by confounding with service type.

**Be careful with correlation.** The heatmap shows how to spot a derived feature, how a sign flip can come from a shared third variable, and why a correlation coefficient measures linear association and nothing about cause.

**Build the baseline first.** Setting a naive reference before any real modeling gives later results a floor to be compared against. It also teaches that a score only has meaning relative to what a trivial strategy achieves.

**Report honestly.** The write-up states where its own tools fall short, such as the log transform that overshoots, the Pearson limits, and the fact that every finding comes from observational data. That habit is as important as the analysis.

## 11. Limitations and next steps

The analysis is descriptive. It uses the full dataset with no train and test split, which is correct for EDA but means the baseline numbers describe this data and not unseen customers. All findings are associations. Several features overlap, so, for example, the effect of payment method cannot be separated from contract type and tenure without a model that adjusts for them.

The first next step is to split the data with stratification, then fit simple models such as logistic regression and a tree-based model, and compare them with this baseline using churn-class precision, recall, F1 and PR-AUC. The second is to handle `TotalCharges` carefully, either by dropping it, transforming it, or replacing it with a feature such as average spend per month, because of its near-duplicate relationship with tenure and monthly charges. The third is to encode the categorical columns properly and keep the "No internet service" levels as valid categories.

## 12. How to run

Open `eda_churn.ipynb` in Google Colab and choose Runtime, then Run all. When the first cell asks for a file, upload `WA_Fn-UseC_-Telco-Customer-Churn.csv`. Colab already provides every library the notebook needs.

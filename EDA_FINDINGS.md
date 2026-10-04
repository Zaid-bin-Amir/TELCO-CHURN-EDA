# Telco Customer Churn: EDA Findings

Prepared by: Zaid

All numbers below come from running `eda_churn.py` on the full dataset (7,043 customers, 21 columns). Figures are in the `figures/` folder.

## 1. Data quality note

`TotalCharges` loads as text because 11 rows contain a blank string. All 11 of those rows have `tenure = 0`, and every `tenure = 0` row is blank. So this is a known data-type quirk (new customers who have not been billed yet), not random missingness. I converted the column to numeric and set those 11 values to 0. No other column has missing values, and there are no duplicate customer IDs.

Several service columns use "No internet service" or "No phone service" as their own category. That is a real answer, not missing data.

## 2. Summary statistics

| Column | Mean | Median | Std | Variance | Skewness |
|---|---|---|---|---|---|
| tenure | 32.37 | 29.00 | 24.56 | 603.17 | 0.24 |
| MonthlyCharges | 64.76 | 70.35 | 30.09 | 905.41 | -0.22 |
| TotalCharges | 2279.73 | 1394.55 | 2266.79 | 5,138,357 | 0.96 |
| SeniorCitizen | 0.162 | 0.00 | 0.369 | 0.136 | 1.83 |

### Skewed distributions and what they imply

1. **TotalCharges is right-skewed (skewness 0.96).** The mean (2,279.73) is about 885 higher than the median (1,394.55) because a long tail of long-tenure, high-bill customers pulls the mean up. The mean overstates what a typical customer has paid, so the median is the better summary. A log transform (`log1p`) removes the right skew, though it overshoots to -0.82, so it is a partial fix. For linear or distance-based models later, this column should be transformed or scaled.
2. **SeniorCitizen is heavily skewed (skewness 1.83).** It is a 0/1 flag, so this skew simply reflects that only 16.2% of customers are senior citizens. The mean (0.162) is just the share of seniors and the median is 0. This is a rare group, so any pattern found for seniors rests on a much smaller sample than for non-seniors.

Two more columns deserve a warning even though their skewness coefficient is small:

- **tenure** (skew 0.24) is U-shaped, with a spike of brand-new customers (21.0% have 6 months or less) and another spike at the 66 to 72 month end (14.7%). The mean of 32.4 describes almost nobody.
- **MonthlyCharges** (skew -0.22) is bimodal, with a large cluster of about 1,400 customers paying roughly 18 to 25 per month and a second wide hump around 70 to 100. A low skewness number hides this, so the plot matters more than the coefficient here.

## 3. Target distribution

| Class | Count | Percent |
|---|---|---|
| No Churn | 5,174 | 73.46% |
| Churn | 1,869 | 26.54% |

The imbalance ratio is about 2.77 to 1. This is moderate imbalance, in line with the 26 to 27% positive class expected for this dataset. The implication for metric choice is that accuracy will be inflated by the majority class, so later models should be judged on precision, recall, F1 for the churn class, and probably PR-AUC or ROC-AUC, with stratified splits. See `BASELINE.md` for a concrete demonstration.

## 4. Bivariate insights (feature vs churn)

### 4.1 Contract type vs churn (`figures/03_contract_vs_churn.png`)

| Contract | Customers | Churn rate |
|---|---|---|
| Month-to-month | 3,875 | 42.71% |
| One year | 1,473 | 11.27% |
| Two year | 1,695 | 2.83% |

Month-to-month customers churn at roughly 15 times the rate of two-year customers and far above the overall 26.5%. They also make up 55% of the customer base, so this one segment accounts for 1,655 of the 1,869 churners (about 89%). The pattern is strong and monotonic: the longer the commitment, the lower the churn. This is an association, not proof that contracts cause retention. Customers who already intend to stay are probably the ones who choose longer contracts, so contract type is partly a signal of existing loyalty.

### 4.2 Tenure vs churn (`figures/04_tenure_vs_churn.png`)

| Tenure band (months) | Customers | Churn rate |
|---|---|---|
| 0 to 6 | 1,481 | 52.94% |
| 7 to 12 | 705 | 35.89% |
| 13 to 24 | 1,024 | 28.71% |
| 25 to 48 | 1,594 | 20.39% |
| 49 to 60 | 832 | 14.42% |
| 61 to 72 | 1,407 | 6.61% |

Churners have a mean tenure of 17.98 months and a median of only 10, against a mean of 37.57 and median of 38 for those who stayed. Churn falls steadily with tenure, from 52.9% in the first six months to 6.6% after five years. The first year is clearly the danger zone. One caution is survivorship: the long-tenure group only contains people who did not leave earlier, so part of the low late churn is a selection effect and not purely "loyalty grows with time".

### 4.3 Monthly charges vs churn (`figures/05_monthly_charges_vs_churn.png`)

| Monthly charge band | Customers | Churn rate |
|---|---|---|
| up to 30 | 1,653 | 9.80% |
| 30 to 50 | 646 | 30.80% |
| 50 to 70 | 1,161 | 20.76% |
| 70 to 90 | 1,844 | 37.80% |
| over 90 | 1,739 | 32.78% |

Churners pay more on average (mean 74.44, median 79.65) than non-churners (mean 61.27, median 64.43). The relationship is not a clean straight line: the cheapest band churns least, the 70 to 90 band churns most, and the 50 to 70 band dips in between. Of the customers paying 30 or less, 92% have no internet service (phone-only), and that group churns at only 7.40%. Fiber optic customers pay about 91.5 per month on average (DSL 58.1, no internet 21.1) and have the highest churn rate in the data (41.89% versus 18.96% for DSL). So the price effect is tangled up with service type and contract, and this chart alone cannot say that price itself drives people away.

### 4.4 Extra views (`figures/06_internet_payment_vs_churn.png`)

Electronic check users churn at 45.29%, compared with 15.24% to 19.11% for the other three payment methods. Fiber optic customers churn at 41.89%. Both are worth carrying into later modeling, with the same caution that they overlap with contract type and tenure.

## 5. Correlation heatmap (`figures/07_correlation_heatmap.png`)

| Pair | Pearson r |
|---|---|
| tenure vs TotalCharges | +0.826 |
| MonthlyCharges vs TotalCharges | +0.651 |
| tenure vs Churn | -0.352 |
| tenure vs MonthlyCharges | +0.248 |
| MonthlyCharges vs SeniorCitizen | +0.220 |
| TotalCharges vs Churn | -0.198 |
| MonthlyCharges vs Churn | +0.193 |
| SeniorCitizen vs Churn | +0.151 |
| TotalCharges vs SeniorCitizen | +0.103 |
| tenure vs SeniorCitizen | +0.017 |

Flags:

- **Strongest relationship, not surprising once checked:** `TotalCharges` is almost a derived column. The correlation between `TotalCharges` and `tenure x MonthlyCharges` is 0.9996, and 94.4% of customers sit within 10% of that product. This is a near-duplicate feature that matters for later feature engineering and any linear model (multicollinearity).
- **Surprising sign flip:** `MonthlyCharges` correlates positively with churn (+0.19) while `TotalCharges` correlates negatively (-0.20). Both are driven by tenure: high monthly bills raise churn, but a high cumulative total simply means the customer has stayed for a long time.
- **Weakest:** `tenure` and `SeniorCitizen` are essentially uncorrelated (+0.017), and senior status has only a modest link to churn (+0.15).
- **Limits of Pearson here:** the strongest link to churn is tenure (-0.35), which is moderate at best. Pearson only measures linear association, so it understates the non-monotonic MonthlyCharges pattern seen above, and it says nothing about causation. Correlations with the binary churn flag are also compressed by the 0/1 coding.

## 6. Takeaways

- Churn is concentrated in new customers on month-to-month contracts, paying by electronic check, with fiber optic service and higher monthly bills.
- Class imbalance (26.5% churn) means accuracy is not a safe headline metric.
- `TotalCharges` is largely redundant with `tenure` and `MonthlyCharges` and should be handled carefully when modeling.
- All findings here are associations from observational data and should not be read as causal.

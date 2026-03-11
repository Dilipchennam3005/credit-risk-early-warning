# Meridian Bank — Loan Dataset Data Dictionary
## Credit Risk Early Warning System
**Version:** 1.0 | **Author:** Dilip Chennam | **Records:** 250,000

---

## File Descriptions

| File | Records | Size | Description |
|------|---------|------|-------------|
| `loans_raw.csv` | 250,000 | ~46 MB | Full combined dataset |
| `loans_mortgage.csv` | 140,000 | ~26 MB | Mortgage loans only |
| `loans_auto.csv` | 110,000 | ~20 MB | Auto loans only |

---

## Field Definitions

### Identifiers
| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `loan_id` | string | MRD-MOR-2021-0000001 | Unique loan identifier. Format: MRD-{type}-{year}-{seq} |
| `loan_type` | string | Mortgage / Auto | Product type |
| `origination_date` | date | 2021-06-15 | Date loan was funded |
| `origination_year` | int | 2021 | Year of origination |
| `origination_quarter` | string | Q2 | Quarter of origination |

### Loan Terms
| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `loan_amount` | float | 285000.00 | Original principal amount ($) |
| `term_months` | int | 360 | Loan term in months (mortgage: 180/240/360; auto: 36/48/60/72/84) |
| `interest_rate` | float | 6.125 | Annual interest rate (%) |
| `loan_purpose` | string | Purchase | Purpose: Purchase / Refinance / Cash-Out Refinance |
| `ltv_ratio` | float | 87.5 | Loan-to-value ratio (%) at origination |
| `monthly_payment` | float | 1842.50 | Estimated monthly payment ($) |

### Borrower Profile
| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `borrower_age` | int | 38 | Borrower age at origination |
| `annual_income` | float | 95000.00 | Gross annual income ($) |
| `monthly_income` | float | 7916.67 | Gross monthly income ($) |
| `employment_years` | float | 5.5 | Years at current employer |
| `is_self_employed` | int | 0 | 1 = self-employed, 0 = W-2 employee |
| `employment_sector` | string | Technology | Industry sector |
| `state` | string | CA | US state of residence (2-letter code) |

### Credit Profile
| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `credit_score` | int | 712 | FICO credit score (300–850) |
| `credit_tier` | string | Good | Tier: Exceptional (800+) / Very Good (740–799) / Good (670–739) / Fair (580–669) / Poor (<580) |
| `dti_ratio` | float | 32.4 | Debt-to-income ratio (%) including proposed payment |
| `num_open_accounts` | int | 8 | Number of open credit accounts |
| `num_delinquencies` | int | 0 | Number of past delinquencies on credit report |
| `num_inquiries_12m` | int | 1 | Hard credit inquiries in past 12 months |

### Economic Context
| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `fed_funds_rate` | float | 5.08 | Federal Funds Rate at origination (%) — affects spread and default probability |

### Mortgage-Specific Fields
| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `property_type` | string | Single Family | Property type: Single Family / Condo / Townhouse / Multi-Family |
| `property_value` | float | 325000.00 | Appraised property value ($) at origination |

### Auto-Specific Fields
| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `vehicle_make` | string | Toyota | Vehicle manufacturer |
| `vehicle_model` | string | Camry | Vehicle model |
| `vehicle_year` | int | 2022 | Model year of vehicle |
| `vehicle_type` | string | New | New / Used / Certified Pre-Owned |

### Performance / Target Variable
| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `loan_status` | string | Current | Current / Paid Off / 30 DPD / 60 DPD / 90+ DPD / Default / Charged Off |
| `is_default` | int | 0 | **Target variable.** 1 = defaulted, 0 = performing |
| `default_probability` | float | 0.0823 | Theoretical default probability used to generate outcome |
| `months_to_default` | int | 14 | Months from origination to default event (null if not defaulted) |

---

## Loan Status Definitions

| Status | Description |
|--------|-------------|
| Current | Loan is active and payments are up to date |
| Paid Off | Loan has been fully repaid |
| 30 DPD | 30 days past due — early delinquency |
| 60 DPD | 60 days past due — serious delinquency |
| 90+ DPD | 90+ days past due — pre-default |
| Default | Borrower has formally defaulted |
| Charged Off | Bank has written off the loan as a loss |

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total records | 250,000 |
| Date range | 2018-01-01 to 2024-12-31 |
| Overall default rate | 11.03% |
| Mortgage default rate | 11.29% |
| Auto default rate | 10.71% |
| Avg loan amount | $140,685 |
| Avg credit score | 692 |
| Avg DTI ratio | 24.2% |
| Avg interest rate | 5.66% |

---

## Economic Regime Notes

The dataset intentionally captures three distinct economic environments:

| Period | Regime | Effect on Data |
|--------|--------|----------------|
| 2018–2019 | Gradual rate normalization (1.4%→2.4%) | Moderate defaults, normal origination volume |
| 2020 Q1 | COVID-19 shock | Spike in defaults, near-zero rates by Q2 |
| 2020 Q2–2021 | Near-zero rates (0.06%–0.09%) | High origination volume, low defaults |
| 2022–2023 | Aggressive rate hikes (0.08%→5.33%) | Rising defaults, stress on variable-rate loans |
| 2024 | Rate stabilization (5.33%→4.83%) | Defaults plateauing |

---

## Class Imbalance Note

The dataset has an ~11% default rate, consistent with real-world mortgage/auto portfolios. ML models must account for this imbalance using techniques such as:
- `class_weight='balanced'` in scikit-learn
- SMOTE oversampling
- Threshold tuning on predicted probabilities
- Evaluation via AUC-ROC and Precision-Recall curves (not accuracy)

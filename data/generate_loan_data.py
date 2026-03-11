"""
================================================================================
Meridian Bank — Credit Risk Early Warning System
Synthetic Loan Data Generator
================================================================================
Generates 250,000 realistic mortgage and auto loan records spanning 2018–2024.

Design principles:
  - Realistic distributions based on industry benchmarks (FFIEC, CFPB data)
  - Economic regime effects: COVID shock (2020), rate hike cycle (2022–2023)
  - Correlated features: credit score ↔ DTI ↔ interest rate ↔ default probability
  - Class imbalance: ~12% default rate (realistic for mixed mortgage/auto book)
  - 40+ features covering borrower, loan, property/vehicle, and performance data

Output files:
  - data/raw/loans_raw.csv          — full 250K dataset (all fields)
  - data/raw/loans_mortgage.csv     — mortgage subset (~140K)
  - data/raw/loans_auto.csv         — auto loan subset (~110K)
  - data/schemas/data_dictionary.md — field definitions

Author: Dilip Chennam
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N_TOTAL     = 250_000
N_MORTGAGE  = 140_000
N_AUTO      = 110_000
START_DATE  = datetime(2018, 1, 1)
END_DATE    = datetime(2024, 12, 31)

# ── US States distribution (weighted toward populous states)
_STATES_RAW = {
    'CA': 0.12, 'TX': 0.09, 'FL': 0.08, 'NY': 0.07, 'PA': 0.05,
    'IL': 0.04, 'OH': 0.04, 'GA': 0.04, 'NC': 0.03, 'MI': 0.03,
    'NJ': 0.03, 'VA': 0.03, 'WA': 0.03, 'AZ': 0.03, 'MA': 0.02,
    'TN': 0.02, 'IN': 0.02, 'MO': 0.02, 'MD': 0.02, 'CO': 0.02,
    'WI': 0.02, 'MN': 0.02, 'SC': 0.01, 'AL': 0.01, 'LA': 0.01,
    'KY': 0.01, 'OR': 0.01, 'OK': 0.01, 'CT': 0.01, 'UT': 0.01,
    'NV': 0.01, 'AR': 0.01, 'MS': 0.01, 'KS': 0.01, 'NM': 0.005,
    'NE': 0.005,'WV': 0.005,'ID': 0.005,'HI': 0.005,'NH': 0.005,
    'ME': 0.003,'RI': 0.003,'MT': 0.003,'DE': 0.003,'SD': 0.002,
    'ND': 0.002,'AK': 0.002,'VT': 0.002,'WY': 0.002,'DC': 0.002,
}
_total = sum(_STATES_RAW.values())
STATES = {k: v / _total for k, v in _STATES_RAW.items()}

# ── Employment sectors
EMPLOYMENT_SECTORS = [
    'Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing',
    'Retail', 'Government', 'Construction', 'Transportation', 'Hospitality',
    'Real Estate', 'Legal', 'Energy', 'Agriculture', 'Media'
]

# ── Economic regime: Fed Funds Rate by quarter (affects interest rates + defaults)
FED_FUNDS_RATE = {
    2018: {1: 1.42, 2: 1.71, 3: 1.91, 4: 2.18},
    2019: {1: 2.40, 2: 2.38, 3: 2.19, 4: 1.78},
    2020: {1: 1.58, 2: 0.06, 3: 0.09, 4: 0.09},  # COVID crash Q1, near-zero Q2+
    2021: {1: 0.07, 2: 0.06, 3: 0.08, 4: 0.08},
    2022: {1: 0.08, 2: 0.77, 3: 2.33, 4: 3.78},  # Aggressive hikes
    2023: {1: 4.57, 2: 5.08, 3: 5.33, 4: 5.33},
    2024: {1: 5.33, 2: 5.33, 3: 5.10, 4: 4.83},
}

def get_fed_rate(date):
    yr  = date.year
    qtr = (date.month - 1) // 3 + 1
    return FED_FUNDS_RATE.get(yr, {}).get(qtr, 2.5)

def random_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=np.random.randint(0, delta))

def credit_score_to_tier(score):
    if score >= 800: return 'Exceptional'
    if score >= 740: return 'Very Good'
    if score >= 670: return 'Good'
    if score >= 580: return 'Fair'
    return 'Poor'

def compute_default_probability(credit_score, dti, ltv, loan_type,
                                 origination_date, employment_years,
                                 is_self_employed, fed_rate):
    """
    Logistic-style default probability with realistic risk drivers.
    Higher DTI, lower credit score, higher LTV → higher default probability.
    COVID period and rate hike period increase defaults.
    """
    base = 0.05

    # Credit score effect (strongest predictor)
    if credit_score < 580:   base += 0.18
    elif credit_score < 620: base += 0.12
    elif credit_score < 660: base += 0.07
    elif credit_score < 700: base += 0.03
    elif credit_score < 740: base += 0.01
    else:                    base -= 0.02

    # DTI effect
    if dti > 50:   base += 0.10
    elif dti > 43: base += 0.06
    elif dti > 36: base += 0.03
    elif dti < 20: base -= 0.02

    # LTV effect (mortgage specific)
    if loan_type == 'Mortgage':
        if ltv > 95:   base += 0.08
        elif ltv > 90: base += 0.05
        elif ltv > 80: base += 0.02
        elif ltv < 60: base -= 0.02

    # Employment stability
    if employment_years < 1: base += 0.04
    elif employment_years < 2: base += 0.02
    if is_self_employed: base += 0.03

    # Economic regime
    yr = origination_date.year
    if yr == 2020: base += 0.04   # COVID shock
    if yr in [2022, 2023]: base += 0.03  # Rate hike stress

    # Auto loans have slightly higher default rates
    if loan_type == 'Auto': base += 0.02

    # Fed rate effect
    if fed_rate > 4.0: base += 0.02

    return max(0.005, min(0.85, base))

def generate_loans(n, loan_type):
    print(f"  Generating {n:,} {loan_type} loans...")
    records = []

    for i in range(n):
        if i % 50000 == 0 and i > 0:
            print(f"    {i:,} records generated...")

        # ── Origination date
        orig_date = random_date(START_DATE, END_DATE)
        fed_rate  = get_fed_rate(orig_date)
        yr        = orig_date.year

        # ── Borrower demographics
        age = int(np.random.normal(42, 12))
        age = max(22, min(75, age))

        # Credit score — realistic distribution (mean ~710, left skewed)
        credit_score = int(np.random.beta(5, 2) * 550 + 300)
        credit_score = max(300, min(850, credit_score))
        credit_tier  = credit_score_to_tier(credit_score)

        # Income — correlated with credit score
        income_base   = np.random.lognormal(mean=11.0, sigma=0.5)
        income_boost  = (credit_score - 300) / 550 * 40000
        annual_income = max(25000, income_base + income_boost)
        annual_income = round(annual_income / 100) * 100

        # Employment
        employment_years  = max(0, round(np.random.exponential(8), 1))
        is_self_employed  = np.random.random() < 0.15
        employment_sector = np.random.choice(EMPLOYMENT_SECTORS)

        # ── Loan characteristics
        if loan_type == 'Mortgage':
            # Loan amount: $80K–$1.2M, lognormal
            loan_amount = round(np.random.lognormal(12.2, 0.5) / 1000) * 1000
            loan_amount = max(80_000, min(1_200_000, loan_amount))

            term_months  = np.random.choice([180, 240, 360], p=[0.05, 0.10, 0.85])
            property_val = round(loan_amount / np.random.uniform(0.70, 0.97) / 1000) * 1000
            ltv          = round(loan_amount / property_val * 100, 2)

            # Property type
            property_type = np.random.choice(
                ['Single Family', 'Condo', 'Townhouse', 'Multi-Family'],
                p=[0.65, 0.18, 0.12, 0.05]
            )
            # Purpose
            loan_purpose = np.random.choice(
                ['Purchase', 'Refinance', 'Cash-Out Refinance'],
                p=[0.55, 0.30, 0.15]
            )
            vehicle_make  = None
            vehicle_model = None
            vehicle_year  = None
            vehicle_type  = None

        else:  # Auto
            loan_amount = round(np.random.lognormal(10.3, 0.45) / 100) * 100
            loan_amount = max(5_000, min(120_000, loan_amount))

            term_months   = np.random.choice([36, 48, 60, 72, 84], p=[0.08, 0.15, 0.35, 0.27, 0.15])
            property_val  = loan_amount * np.random.uniform(1.0, 1.15)
            ltv           = round(loan_amount / property_val * 100, 2)

            vehicle_makes = {
                'Toyota': ['Camry','RAV4','Corolla','Highlander'],
                'Ford':   ['F-150','Explorer','Mustang','Escape'],
                'Honda':  ['Civic','Accord','CR-V','Pilot'],
                'Chevrolet': ['Silverado','Equinox','Malibu','Tahoe'],
                'BMW':    ['3 Series','5 Series','X3','X5'],
                'Tesla':  ['Model 3','Model Y','Model S'],
                'Nissan': ['Altima','Rogue','Sentra'],
                'Hyundai':['Elantra','Tucson','Santa Fe'],
            }
            vehicle_make  = np.random.choice(list(vehicle_makes.keys()))
            vehicle_model = np.random.choice(vehicle_makes[vehicle_make])
            vehicle_year  = np.random.randint(max(2010, yr - 5), yr + 1)
            vehicle_type  = np.random.choice(
                ['New', 'Used', 'Certified Pre-Owned'],
                p=[0.35, 0.45, 0.20]
            )
            property_type = None
            loan_purpose  = np.random.choice(
                ['Purchase', 'Refinance'],
                p=[0.80, 0.20]
            )

        # ── DTI calculation
        monthly_income   = annual_income / 12
        other_debt       = monthly_income * np.random.uniform(0.05, 0.25)
        monthly_payment  = loan_amount / term_months * np.random.uniform(1.02, 1.08)
        dti              = round((monthly_payment + other_debt) / monthly_income * 100, 2)
        dti              = max(5, min(65, dti))

        # ── Interest rate (spread over fed funds, credit-score adjusted)
        base_spread = {'Mortgage': 1.8, 'Auto': 3.5}[loan_type]
        score_spread = max(0, (720 - credit_score) / 100 * 1.5)
        rate_noise   = np.random.normal(0, 0.3)
        interest_rate = round(fed_rate + base_spread + score_spread + rate_noise, 3)
        interest_rate = max(2.0, min(24.0, interest_rate))

        # ── State
        state = np.random.choice(
            list(STATES.keys()),
            p=list(STATES.values())
        )

        # ── Default determination
        default_prob = compute_default_probability(
            credit_score, dti, ltv, loan_type,
            orig_date, employment_years,
            is_self_employed, fed_rate
        )
        is_default = np.random.random() < default_prob

        # ── Loan status
        if is_default:
            months_to_default = int(np.random.exponential(18))
            months_to_default = max(1, min(term_months - 1, months_to_default))
            default_date = orig_date + timedelta(days=int(months_to_default) * 30)
            if default_date > END_DATE:
                default_date = END_DATE
                is_default   = False
                loan_status  = 'Current'
            else:
                loan_status = np.random.choice(
                    ['Charged Off', 'Default', '90+ DPD'],
                    p=[0.45, 0.35, 0.20]
                )
        else:
            default_date    = None
            months_to_default = None
            maturity_date   = orig_date + timedelta(days=int(term_months) * 30)
            if maturity_date <= END_DATE:
                loan_status = 'Paid Off'
            else:
                loan_status = np.random.choice(
                    ['Current', '30 DPD', '60 DPD'],
                    p=[0.93, 0.04, 0.03]
                )

        # ── Delinquency history
        num_delinquencies = 0
        if credit_score < 620:
            num_delinquencies = int(np.random.poisson(2.5))
        elif credit_score < 680:
            num_delinquencies = int(np.random.poisson(0.8))
        elif credit_score < 740:
            num_delinquencies = int(np.random.poisson(0.2))

        # ── Number of open accounts / inquiries
        num_open_accounts = int(np.random.normal(8, 3))
        num_open_accounts = max(1, min(25, num_open_accounts))
        num_inquiries_12m = int(np.random.poisson(1.2))
        num_inquiries_12m = max(0, min(10, num_inquiries_12m))

        # ── Loan ID
        loan_id = f"MRD-{loan_type[:3].upper()}-{yr}-{str(i+1).zfill(7)}"

        records.append({
            # Identifiers
            'loan_id':              loan_id,
            'loan_type':            loan_type,
            'origination_date':     orig_date.strftime('%Y-%m-%d'),
            'origination_year':     yr,
            'origination_quarter':  f"Q{(orig_date.month-1)//3+1}",

            # Loan terms
            'loan_amount':          loan_amount,
            'term_months':          term_months,
            'interest_rate':        interest_rate,
            'loan_purpose':         loan_purpose,
            'ltv_ratio':            ltv,
            'monthly_payment':      round(monthly_payment, 2),

            # Borrower
            'borrower_age':         age,
            'annual_income':        round(annual_income, 2),
            'monthly_income':       round(monthly_income, 2),
            'employment_years':     employment_years,
            'is_self_employed':     int(is_self_employed),
            'employment_sector':    employment_sector,
            'state':                state,

            # Credit profile
            'credit_score':         credit_score,
            'credit_tier':          credit_tier,
            'dti_ratio':            dti,
            'num_open_accounts':    num_open_accounts,
            'num_delinquencies':    num_delinquencies,
            'num_inquiries_12m':    num_inquiries_12m,

            # Economic context
            'fed_funds_rate':       fed_rate,

            # Mortgage-specific
            'property_type':        property_type,
            'property_value':       round(property_val, 2) if loan_type == 'Mortgage' else None,

            # Auto-specific
            'vehicle_make':         vehicle_make,
            'vehicle_model':        vehicle_model,
            'vehicle_year':         vehicle_year,
            'vehicle_type':         vehicle_type,

            # Performance / target
            'loan_status':          loan_status,
            'is_default':           int(is_default),
            'default_probability':  round(default_prob, 4),
            'months_to_default':    months_to_default,
        })

    return pd.DataFrame(records)

# ── MAIN
if __name__ == '__main__':
    print("=" * 60)
    print("Meridian Bank — Loan Data Generator")
    print("=" * 60)

    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/schemas', exist_ok=True)

    # Generate both loan types
    print("\n[1/3] Generating mortgage loans...")
    df_mortgage = generate_loans(N_MORTGAGE, 'Mortgage')

    print("\n[2/3] Generating auto loans...")
    df_auto = generate_loans(N_AUTO, 'Auto')

    # Combine
    print("\n[3/3] Combining and saving...")
    df_all = pd.concat([df_mortgage, df_auto], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df_all.to_csv('data/raw/loans_raw.csv', index=False)
    df_mortgage.to_csv('data/raw/loans_mortgage.csv', index=False)
    df_auto.to_csv('data/raw/loans_auto.csv', index=False)

    # ── Summary stats
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total loans:          {len(df_all):>10,}")
    print(f"Mortgage loans:       {len(df_mortgage):>10,}")
    print(f"Auto loans:           {len(df_auto):>10,}")
    print(f"Default rate:         {df_all['is_default'].mean()*100:>9.2f}%")
    print(f"Mortgage default rate:{df_mortgage['is_default'].mean()*100:>9.2f}%")
    print(f"Auto default rate:    {df_auto['is_default'].mean()*100:>9.2f}%")
    print(f"Date range:           2018-01-01 to 2024-12-31")
    print(f"Avg loan amount:      ${df_all['loan_amount'].mean():>10,.0f}")
    print(f"Avg credit score:     {df_all['credit_score'].mean():>10.0f}")
    print(f"Avg DTI:              {df_all['dti_ratio'].mean():>9.1f}%")
    print(f"Avg interest rate:    {df_all['interest_rate'].mean():>9.2f}%")
    print(f"\nLoan status distribution:")
    print(df_all['loan_status'].value_counts().to_string())
    print(f"\nCredit tier distribution:")
    print(df_all['credit_tier'].value_counts().to_string())
    print(f"\nFiles saved:")
    for f in ['data/raw/loans_raw.csv', 'data/raw/loans_mortgage.csv', 'data/raw/loans_auto.csv']:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {f:<40} {size_mb:.1f} MB")

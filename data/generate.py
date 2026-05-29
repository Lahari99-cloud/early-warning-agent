"""
Synthetic data generator for attrition early-warning agent.
"""
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from .schema import EmployeeFeatures
import os

# Seed for reproducibility
SEED = 42

# Reference date for "today" (from CLAUDE.md currentDate)
TODAY = pd.Timestamp('2026-05-29')

def generate_synthetic_data(n_employees: int = 1500) -> pd.DataFrame:
    """
    Generate synthetic employee data for attrition modeling.

    Args:
        n_employees: Number of employee records to generate.

    Returns:
        DataFrame with synthetic employee data.
    """
    # Create and seed Faker instance for this call to ensure reproducibility
    fake = Faker()
    Faker.seed(SEED)
    fake.seed_instance(SEED)
    np.random.seed(SEED)

    # Generate base features
    employee_ids = np.arange(n_employees)

    # Personal info
    first_names = [fake.first_name() for _ in range(n_employees)]
    last_names = [fake.last_name() for _ in range(n_employees)]

    # Department and job title
    departments = ['Engineering', 'Sales', 'HR', 'Finance', 'Marketing']
    dept_choices = np.random.choice(departments, n_employees, p=[0.4, 0.2, 0.1, 0.15, 0.15])

    # Job titles per department (simplified)
    job_title_map = {
        'Engineering': ['Software Engineer', 'Senior Engineer', 'Tech Lead', 'Engineering Manager'],
        'Sales': ['Sales Rep', 'Senior Sales Rep', 'Sales Manager', 'Regional Sales Director'],
        'HR': ['HR Coordinator', 'HR Generalist', 'HR Manager', 'HR Director'],
        'Finance': ['Financial Analyst', 'Senior Financial Analyst', 'Finance Manager', 'Finance Director'],
        'Marketing': ['Marketing Specialist', 'Marketing Manager', 'Director of Marketing', 'VP Marketing']
    }
    job_titles = []
    for dept in dept_choices:
        job_titles.append(np.random.choice(job_title_map[dept]))

    # Tenure and hire date
    years_at_company = np.random.uniform(0, 5, n_employees)  # 0-5 years
    hire_date = TODAY - pd.to_timedelta(years_at_company * 365, unit='D')

    # Level (L1-L5) with distribution skewed to lower levels
    level_probs = [0.3, 0.25, 0.2, 0.15, 0.1]
    levels = np.random.choice(['L1', 'L2', 'L3', 'L4', 'L5'], n_employees, p=level_probs)

    # Salary band base by level
    level_base_salary = {
        'L1': 50000,
        'L2': 70000,
        'L3': 90000,
        'L4': 120000,
        'L5': 150000
    }
    base_salary = np.array([level_base_salary[lev] for lev in levels])

    # Compensation vs band percentage (80-120% typical, but we'll allow wider range for risk)
    comp_vs_band_pct = np.random.normal(100, 15, n_employees)  # mean 100%, std 15%
    comp_vs_band_pct = np.clip(comp_vs_band_pct, 70, 130)  # reasonable bounds

    # Actual salary
    salary = base_salary * (comp_vs_band_pct / 100.0)

    # Months since promotion (0-36 months, but constrained by tenure)
    max_months = years_at_company * 12
    months_since_promo = np.zeros(n_employees, dtype=int)
    for i in range(n_employees):
        max_m = int(max_months[i])
        if max_m > 0:
            months_since_promo[i] = np.random.randint(0, max_m + 1)
        else:
            months_since_promo[i] = 0

    # Last promotion date derived from hire date and months since promo
    # Use proper date offset for months
    last_promotion_date = []
    for i in range(n_employees):
        promo_date = hire_date[i] + pd.DateOffset(months=int(max_months[i] - months_since_promo[i]))
        last_promotion_date.append(promo_date)
    last_promotion_date = pd.Series(last_promotion_date)

    # Performance trend: -1 (declining) to +1 (improving)
    perf_trend = np.random.uniform(-1, 1, n_employees)

    # Engagement score (0-100)
    engagement_score = np.random.uniform(0, 100, n_employees)

    # PTO usage percentage (0-100%)
    pto_usage = np.random.uniform(0, 100, n_employees)

    # Manager ID (ensure no self-management)
    manager_id = np.zeros(n_employees, dtype=int)
    for i in range(n_employees):
        # Choose a random manager that is not the employee themselves
        possible_managers = np.setdiff1d(employee_ids, [employee_ids[i]])
        manager_id[i] = np.random.choice(possible_managers)

    # One-on-one frequency (meetings per month, 0-4)
    one_on_one_freq = np.random.uniform(0, 4, n_employees)

    # Internal moves (number of internal transfers)
    internal_moves = np.random.choice([0, 1, 2, 3], n_employees, p=[0.5, 0.3, 0.15, 0.05])

    # Attrition risk label generation (~12% positive)
    # Risk factors: low comp_vs_band_pct, high perf_trend, long months_since_promo
    # Normalize each factor to 0-1 scale where higher means higher risk
    comp_risk = (130 - comp_vs_band_pct) / (130 - 70)  # 0 when comp=130, 1 when comp=70
    perf_risk = (perf_trend + 1) / 2  # 0 when perf_trend=-1, 1 when perf_trend=+1
    time_risk = months_since_promo / np.maximum(max_months, 1)  # proportion of tenure since promo

    # Combine risks with weights
    risk_score = (0.4 * comp_risk + 0.3 * perf_risk + 0.3 * time_risk)

    # Determine threshold for ~12% positive rate
    threshold = np.percentile(risk_score, 88)  # top 12% are attrition risk
    attrition_risk = (risk_score >= threshold).astype(int)  # binary label

    # Generate exit note and survey blurb for all employees
    exit_notes = [fake.sentence(nb_words=6) if attrition_risk[i] == 1 else ""
                  for i in range(n_employees)]
    survey_blurbs = [fake.sentence(nb_words=10) for _ in range(n_employees)]

    # Create DataFrame
    df = pd.DataFrame({
        'employee_id': employee_ids,
        'first_name': first_names,
        'last_name': last_names,
        'department': dept_choices,
        'job_title': job_titles,
        'hire_date': hire_date,
        'salary': salary,
        'engagement_score': engagement_score,
        'last_promotion_date': last_promotion_date,
        'years_at_company': years_at_company,
        'manager_id': manager_id,
        'attrition_risk': attrition_risk,  # binary 0/1 as per labeling instruction
        'level': levels,
        'comp_vs_band_pct': comp_vs_band_pct,
        'months_since_promo': months_since_promo,
        'perf_trend': perf_trend,
        'pto_usage': pto_usage,
        'one_on_one_freq': one_on_one_freq,
        'internal_moves': internal_moves,
        'exit_note': exit_notes,
        'survey_blurb': survey_blurbs
    })

    return df

def save_data_to_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to parquet file.

    Args:
        df: DataFrame to save.
        path: File path for the parquet file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def load_data_from_parquet(path: str) -> pd.DataFrame:
    """
    Load DataFrame from parquet file.

    Args:
        path: File path for the parquet file.

    Returns:
        Loaded DataFrame.
    """
    return pd.read_parquet(path)
"""
Schema definition for synthetic employee data.
"""

from dataclasses import dataclass, fields
import datetime
from typing import ClassVar, Dict, Any


@dataclass
class EmployeeFeatures:
    """Dataclass representing employee features for attrition modeling."""
    employee_id: int
    first_name: str
    last_name: str
    department: str
    job_title: str
    hire_date: datetime.datetime
    salary: float
    engagement_score: float
    last_promotion_date: datetime.datetime
    years_at_company: float
    manager_id: int
    attrition_risk: float  # target variable (0 to 1)
    # Additional features as per instructions
    level: str
    comp_vs_band_pct: float
    months_since_promo: int
    perf_trend: float
    pto_usage: float
    one_on_one_freq: float
    internal_moves: int
    exit_note: str
    survey_blurb: str

    # Class variable for schema backward compatibility
    EMPLOYEE_SCHEMA: ClassVar[Dict[str, str]] = None

    def __post_init__(self):
        # Initialize EMPLOYEE_SCHEMA if not already set
        if EmployeeFeatures.EMPLOYEE_SCHEMA is None:
            EmployeeFeatures.EMPLOYEE_SCHEMA = {
                f.name: (
                    "int" if f.type == int else
                    "float" if f.type == float else
                    "string" if f.type == str else
                    "datetime" if f.type == datetime.datetime else
                    str(f.type)
                )
                for f in fields(self)
            }


# For backward compatibility, create the schema dictionary
# This will be populated when the first EmployeeFeatures instance is created
EMPLOYEE_SCHEMA: Dict[str, str] = {}
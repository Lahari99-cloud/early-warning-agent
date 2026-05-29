"""
Schema definition for synthetic employee data.
"""

# Define the schema for the synthetic employee data
EMPLOYEE_SCHEMA = {
    "employee_id": "int",
    "first_name": "string",
    "last_name": "string",
    "department": "string",
    "job_title": "string",
    "hire_date": "datetime",
    "salary": "float",
    "engagement_score": "float",
    "last_promotion_date": "datetime",
    "years_at_company": "float",
    "manager_id": "int",
    "attrition_risk": "float",  # target variable (0 to 1)
    # Additional features can be added as needed
}
import pandas as pd
import numpy as np
import re
import ast
from typing import List
from langchain.chat_models import ChatOpenAI

SUSPICIOUS_PATTERNS = [
    r"(?i)ignore\s+previous\s+instructions",
    r"(?i)act\s+as\s+(an?\s+)?(admin|hacker|expert)",
    r"(?i)please\s+delete",
    r"(?i)execute\s+this\s+code",
    r"(?i)openai\.com|chatgpt|prompt injection",
]

def detect_prompt_injection(df: pd.DataFrame, sample_size: int = 500) -> List[str]:
    suspicious = []
    sample = df.astype(str).sample(min(len(df), sample_size), random_state=1)
    for col in sample.columns:
        for pattern in SUSPICIOUS_PATTERNS:
            matches = sample[col].str.contains(pattern, na=False, regex=True)
            if matches.any():
                suspicious.append(f"⚠️ Potential prompt injection pattern found in column '{col}'")
                break
    return suspicious

def validate_csv(df: pd.DataFrame) -> List[str]:
    issues = []
    if df.empty:
        issues.append("Uploaded CSV is empty.")
    if df.shape[1] < 2:
        issues.append("CSV should contain at least two columns.")
    if df.shape[0] < 3:
        issues.append("CSV should contain at least three rows.")
    return issues

def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    targets = ["N/A", "n/a", "not available", "Not Available", "none", "None", "not a date", ""]
    return df.replace(targets, np.nan)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return df.convert_dtypes()

def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how='all')

def drop_columns_with_80perc_nulls(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    return df.loc[:, df.isnull().mean() <= threshold]

def standardize_booleans(df: pd.DataFrame) -> pd.DataFrame:
    bool_map = {"yes": True, "no": False, "1": True, "0": False}
    for col in df.columns:
        if df[col].astype(str).str.lower().isin(bool_map.keys()).any():
            df[col] = df[col].astype(str).str.lower().map(bool_map).fillna(df[col])
    return df

TOOLS = {
    "remove_empty_rows": remove_empty_rows,
    "normalize_missing_values": normalize_missing_values,
    "standardize_column_names": standardize_column_names,
    "remove_duplicates": remove_duplicates,
    "convert_dtypes": convert_dtypes,
    "drop_columns_with_80perc_nulls": drop_columns_with_80perc_nulls,
    "standardize_booleans": standardize_booleans
}

def verify_tool_effect(before_df: pd.DataFrame, after_df: pd.DataFrame, tool_name: str) -> bool:
    if tool_name == "remove_empty_rows":
        return len(after_df) < len(before_df)
    elif tool_name == "drop_columns_with_80perc_nulls":
        return after_df.shape[1] < before_df.shape[1]
    elif tool_name == "remove_duplicates":
        return len(after_df) < len(before_df)
    elif tool_name == "standardize_column_names":
        return not before_df.columns.equals(after_df.columns)
    elif tool_name == "standardize_booleans":
        return not before_df.equals(after_df)
    elif tool_name == "convert_dtypes":
        return any(before_df.dtypes != after_df.dtypes)
    elif tool_name == "normalize_missing_values":
        return (before_df.replace(["N/A", "n/a", "not available", "Not Available", "none", "None", "not a date", ""], np.nan).isnull().sum().sum()
                != before_df.isnull().sum().sum())
    return True

def generate_column_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for col in df.columns:
        data = df[col]
        col_summary = {
            "Column": col,
            "Type": str(data.dtype),
            "Non-Null Count": data.notnull().sum(),
            "Missing Values": data.isnull().sum(),
            "Missing (%)": round(data.isnull().mean() * 100, 2),
            "Unique": data.nunique(),
            "Min": data.min() if pd.api.types.is_numeric_dtype(data) else "",
            "Max": data.max() if pd.api.types.is_numeric_dtype(data) else "",
            "Sample Values": ', '.join(map(str, data.dropna().unique()[:5])),
            "Num Outliers": "",
        }
        if pd.api.types.is_numeric_dtype(data):
            try:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()
                col_summary["Num Outliers"] = int(outliers)
            except:
                col_summary["Num Outliers"] = "Error"
        summary.append(col_summary)
    return pd.DataFrame(summary)

def suggest_fixes(df: pd.DataFrame) -> List[str]:
    instruction = f"""
You are a data cleaning assistant. Here's a preview of the dataset:
{df.head().to_string()}

Suggest a list of likely cleaning steps from this set: {list(TOOLS.keys())}.
Respond with a Python list of tool names only.
"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    response = llm.invoke(instruction).content.strip()
    try:
        return ast.literal_eval(response)
    except:
        return []
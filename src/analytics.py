import pandas as pd


def format_number(value, decimals=2):

    if pd.isna(value):
        return "Not available"

    return f"{value:,.{decimals}f}"


def dataset_averages(df):

    columns = [
        "bmi_mean",
        "avg_heart_rate_mean",
        "steps_mean",
        "sleep_hours_mean",
        "water_intake_l_mean",
        "risk_probability",
        "health_score"
    ]

    averages = {}

    for column in columns:

        if column in df.columns:

            averages[column] = round(
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).mean(),
                2
            )

    return averages


def answer_average_bmi(df):

    value = pd.to_numeric(
        df["bmi_mean"],
        errors="coerce"
    ).mean()

    return (
        f"📊 **Average BMI:** "
        f"{format_number(value)}\n\n"
        f"Calculated from {len(df):,} users."
    )


def answer_average_sleep(df):

    value = pd.to_numeric(
        df["sleep_hours_mean"],
        errors="coerce"
    ).mean()

    return (
        f"😴 **Average Sleep:** "
        f"{format_number(value)} hours\n\n"
        f"Calculated from {len(df):,} users."
    )


def answer_average_steps(df):

    value = pd.to_numeric(
        df["steps_mean"],
        errors="coerce"
    ).mean()

    return (
        f"🚶 **Average Daily Steps:** "
        f"{format_number(value, 0)}\n\n"
        f"Calculated from {len(df):,} users."
    )


def answer_average_heart_rate(df):

    value = pd.to_numeric(
        df["avg_heart_rate_mean"],
        errors="coerce"
    ).mean()

    return (
        f"❤️ **Average Heart Rate:** "
        f"{format_number(value)} BPM\n\n"
        f"Calculated from {len(df):,} users."
    )


def answer_average_water(df):

    value = pd.to_numeric(
        df["water_intake_l_mean"],
        errors="coerce"
    ).mean()

    return (
        f"💧 **Average Water Intake:** "
        f"{format_number(value)} liters\n\n"
        f"Calculated from {len(df):,} users."
    )


def answer_highest_risk(df, top_n=10):

    required = [
        "user_id",
        "risk_probability",
        "health_score",
        "health_tier"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        return (
            "Required columns are missing: "
            + ", ".join(missing)
        )

    result = df.copy()

    result["risk_probability"] = pd.to_numeric(
        result["risk_probability"],
        errors="coerce"
    )

    result["health_score"] = pd.to_numeric(
        result["health_score"],
        errors="coerce"
    )

    result = result.dropna(
        subset=["risk_probability"]
    )

    result = result.sort_values(
        "risk_probability",
        ascending=False
    ).head(top_n)

    output = result[
        [
            "user_id",
            "risk_probability",
            "health_score",
            "health_tier"
        ]
    ].copy()

    output["risk_probability"] = (
        output["risk_probability"].round(4)
    )

    output["health_score"] = (
        output["health_score"].round(2)
    )

    return output


def answer_healthiest(df, top_n=10):

    required = [
        "user_id",
        "health_score",
        "health_tier"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        return (
            "Required columns are missing: "
            + ", ".join(missing)
        )

    result = df.copy()

    result["health_score"] = pd.to_numeric(
        result["health_score"],
        errors="coerce"
    )

    result = result.dropna(
        subset=["health_score"]
    )

    result = result.sort_values(
        "health_score",
        ascending=False
    ).head(top_n)

    return result[
        [
            "user_id",
            "health_score",
            "health_tier"
        ]
    ].copy()

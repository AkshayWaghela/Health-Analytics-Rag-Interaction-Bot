import streamlit as st
import pandas as pd

from src.data_loader import load_all_data

from src.rag import HealthRAG

from src.analytics import (
    answer_average_bmi,
    answer_average_sleep,
    answer_average_steps,
    answer_average_heart_rate,
    answer_average_water,
    answer_highest_risk,
    answer_healthiest
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Health Analytics RAG",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background: #fafafa;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def initialize_rag():

    data = load_all_data()

    rag = HealthRAG(
        df=data["df"],
        documents=data["documents"],
        embeddings=data["embeddings"],
        index=data["index"]
    )

    return data, rag


with st.spinner(
    "Loading health analytics system..."
):

    data, rag = initialize_rag()


df = data["df"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 Health Analytics RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about users, health metrics,
    risk probability and dataset-level statistics.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Dataset")

    st.metric(
        "Total Users",
        f"{len(df):,}"
    )

    st.divider()

    st.subheader("Available Analytics")

    st.write(
        "• Average BMI"
    )

    st.write(
        "• Average Sleep"
    )

    st.write(
        "• Average Steps"
    )

    st.write(
        "• Average Heart Rate"
    )

    st.write(
        "• Average Water Intake"
    )

    st.write(
        "• Highest Risk Users"
    )

    st.write(
        "• Healthiest Users"
    )

    st.divider()

    st.caption(
        "Powered by Pandas + FAISS + "
        "Sentence Transformers + Qwen"
    )


# ============================================================
# DASHBOARD
# ============================================================

st.subheader("📈 Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    value = pd.to_numeric(
        df["bmi_mean"],
        errors="coerce"
    ).mean()

    st.metric(
        "Average BMI",
        f"{value:.2f}"
    )


with col2:

    value = pd.to_numeric(
        df["sleep_hours_mean"],
        errors="coerce"
    ).mean()

    st.metric(
        "Average Sleep",
        f"{value:.2f} h"
    )


with col3:

    value = pd.to_numeric(
        df["steps_mean"],
        errors="coerce"
    ).mean()

    st.metric(
        "Average Steps",
        f"{value:,.0f}"
    )


with col4:

    value = pd.to_numeric(
        df["avg_heart_rate_mean"],
        errors="coerce"
    ).mean()

    st.metric(
        "Avg Heart Rate",
        f"{value:.2f} BPM"
    )


with col5:

    value = pd.to_numeric(
        df["water_intake_l_mean"],
        errors="coerce"
    ).mean()

    st.metric(
        "Water Intake",
        f"{value:.2f} L"
    )


st.divider()


# ============================================================
# QUICK ANALYTICS
# ============================================================

st.subheader("⚡ Quick Analytics")

col1, col2 = st.columns(2)


with col1:

    if st.button(
        "🔴 Show Highest Risk Users",
        use_container_width=True
    ):

        result = answer_highest_risk(
            df
        )

        if isinstance(
            result,
            pd.DataFrame
        ):

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.error(result)


with col2:

    if st.button(
        "🟢 Show Healthiest Users",
        use_container_width=True
    ):

        result = answer_healthiest(
            df
        )

        if isinstance(
            result,
            pd.DataFrame
        ):

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.error(result)


st.divider()


# ============================================================
# CHAT
# ============================================================

st.subheader("💬 Ask the Health Assistant")


if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


prompt = st.chat_input(
    "Ask something about the health dataset..."
)


if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)


    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing dataset..."
        ):

            message_lower = prompt.lower()

            # ---------------------------------------------
            # DETERMINISTIC QUESTIONS
            # ---------------------------------------------

            if (
                "average bmi" in message_lower
                or "mean bmi" in message_lower
            ):

                answer = answer_average_bmi(
                    df
                )


            elif (
                "average sleep" in message_lower
                or "mean sleep" in message_lower
            ):

                answer = answer_average_sleep(
                    df
                )


            elif (
                "average steps" in message_lower
                or "mean steps" in message_lower
            ):

                answer = answer_average_steps(
                    df
                )


            elif (
                "average heart rate"
                in message_lower
                or
                "mean heart rate"
                in message_lower
            ):

                answer = answer_average_heart_rate(
                    df
                )


            elif (
                "average water"
                in message_lower
                or
                "water intake average"
                in message_lower
            ):

                answer = answer_average_water(
                    df
                )


            elif (
                "highest risk"
                in message_lower
                or
                "riskiest users"
                in message_lower
                or
                "most risky users"
                in message_lower
            ):

                result = answer_highest_risk(
                    df
                )

                if isinstance(
                    result,
                    pd.DataFrame
                ):

                    st.dataframe(
                        result,
                        use_container_width=True,
                        hide_index=True
                    )

                    answer = (
                        "Here are the "
                        "10 users with the "
                        "highest risk probability."
                    )

                else:

                    answer = result


            elif (
                "healthiest"
                in message_lower
                or
                "highest health score"
                in message_lower
            ):

                result = answer_healthiest(
                    df
                )

                if isinstance(
                    result,
                    pd.DataFrame
                ):

                    st.dataframe(
                        result,
                        use_container_width=True,
                        hide_index=True
                    )

                    answer = (
                        "Here are the "
                        "10 healthiest users "
                        "by health score."
                    )

                else:

                    answer = result


            else:

                answer = rag.answer_user_question(
                    prompt
                )


        st.markdown(answer)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

"""
Clinical Trial Screening - Ultimate Edition

Features:
- Fast 2-step workflow
- Batch processing with CSV upload (patients + protocols)
- Developer mode for JSON access
- Trial history with persistence
- Clean professional UI
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import hashlib

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env only if exists (local dev)
from dotenv import load_dotenv
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)


# =============================================================================
# DEVELOPER AUTHENTICATION
# =============================================================================

DEVELOPER_CREDENTIALS = {
    "david": hashlib.sha256("clinicaltrial".encode()).hexdigest(),
    "melea": hashlib.sha256("clinicaltrial".encode()).hexdigest(),
}


def check_developer_login(username: str, password: str) -> bool:
    """Verify developer credentials."""
    if username.lower() in DEVELOPER_CREDENTIALS:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return DEVELOPER_CREDENTIALS[username.lower()] == password_hash
    return False


def is_developer_mode() -> bool:
    """Check if user is logged in as developer."""
    return st.session_state.get("developer_logged_in", False)


# =============================================================================
# DATABASE - TRIAL HISTORY PERSISTENCE
# =============================================================================

def get_supabase_client():
    """Get Supabase client for database operations."""
    try:
        from supabase import create_client
        url = None
        key = None
        try:
            url = st.secrets.get("supabase", {}).get("url")
            key = st.secrets.get("supabase", {}).get("key")
        except Exception:
            pass
        if not url:
            url = os.getenv("SUPABASE_URL")
        if not key:
            key = os.getenv("SUPABASE_KEY")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None


def save_trial_to_db(trial_id: str, protocol: str = None, result: dict = None):
    """Save trial to database for history."""
    client = get_supabase_client()
    if client:
        try:
            data = {
                "trial_id": trial_id,
                "protocol_text": protocol[:5000] if protocol else None,
                "screening_result": json.dumps(result) if result else None,
                "created_at": datetime.now().isoformat()
            }
            client.table("trial_history").upsert(data, on_conflict="trial_id").execute()
            return True
        except Exception:
            pass

    # Fallback to session state
    if "trial_history_local" not in st.session_state:
        st.session_state.trial_history_local = []
    existing_ids = [t["trial_id"] for t in st.session_state.trial_history_local]
    if trial_id not in existing_ids:
        st.session_state.trial_history_local.append({
            "trial_id": trial_id,
            "created_at": datetime.now().isoformat(),
            "has_result": result is not None
        })
        st.session_state.trial_history_local = st.session_state.trial_history_local[-50:]
    return False


def get_trial_history() -> List[dict]:
    """Get trial history from database or session."""
    client = get_supabase_client()
    if client:
        try:
            response = client.table("trial_history").select("trial_id, created_at").order("created_at", desc=True).limit(50).execute()
            return response.data
        except Exception:
            pass
    return st.session_state.get("trial_history_local", [])


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

def get_api_key() -> str:
    """Get Google API key."""
    try:
        api_key = st.secrets.get("google", {}).get("api_key")
        if api_key and api_key != "your_gemini_api_key_here":
            os.environ["GOOGLE_API_KEY"] = api_key
            return api_key
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if api_key and api_key != "your_gemini_api_key_here":
            os.environ["GOOGLE_API_KEY"] = api_key
            return api_key
    except Exception:
        pass
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        return api_key
    if "google_api_key" in st.session_state and st.session_state.google_api_key:
        os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
        return st.session_state.google_api_key
    return None


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Clinical Trial Screening",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    .eligibility-eligible {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white; padding: 20px; border-radius: 10px;
        text-align: center; font-size: 28px; font-weight: bold;
    }
    .eligibility-ineligible {
        background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
        color: white; padding: 20px; border-radius: 10px;
        text-align: center; font-size: 28px; font-weight: bold;
    }
    .eligibility-uncertain {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: black; padding: 20px; border-radius: 10px;
        text-align: center; font-size: 28px; font-weight: bold;
    }
    .api-status {
        display: flex; align-items: center; gap: 8px;
        padding: 5px 10px; border-radius: 5px; font-size: 12px;
    }
    .api-status-ok { background: #d4edda; color: #155724; }
    .api-status-error { background: #f8d7da; color: #721c24; }
    .green-dot {
        width: 10px; height: 10px; border-radius: 50%;
        background: #28a745; display: inline-block;
    }
    .red-dot {
        width: 10px; height: 10px; border-radius: 50%;
        background: #dc3545; display: inline-block;
    }
    .dev-badge {
        background: #6f42c1; color: white;
        padding: 3px 8px; border-radius: 10px; font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "screening_result" not in st.session_state:
    st.session_state.screening_result = None
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "patient_validated" not in st.session_state:
    st.session_state.patient_validated = False
if "trial_history_local" not in st.session_state:
    st.session_state.trial_history_local = []
if "selected_trial_id" not in st.session_state:
    st.session_state.selected_trial_id = ""
if "developer_logged_in" not in st.session_state:
    st.session_state.developer_logged_in = False
if "uploaded_protocols" not in st.session_state:
    st.session_state.uploaded_protocols = {}
if "batch_patients" not in st.session_state:
    st.session_state.batch_patients = []


def clear_session():
    """Clear all session state for new patient."""
    st.session_state.screening_result = None
    st.session_state.patient_data = None
    st.session_state.patient_validated = False


def validate_patient_data(data: dict) -> tuple:
    """Validate patient data structure (backend validation)."""
    required_fields = ["patient_id", "age", "sex"]
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    if not isinstance(data.get("age"), (int, float)) or data["age"] < 0:
        return False, "Invalid age value"
    if data.get("sex") not in ["male", "female", "other"]:
        return False, "Invalid sex value"
    return True, "Valid"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_eligibility_class(decision: str) -> str:
    return f"eligibility-{decision.lower()}"


def create_confidence_gauge(confidence: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#6f42c1"},
            'steps': [
                {'range': [0, 70], 'color': "#dc3545"},
                {'range': [70, 80], 'color': "#ffc107"},
                {'range': [80, 90], 'color': "#17a2b8"},
                {'range': [90, 100], 'color': "#28a745"}
            ],
        }
    ))
    fig.update_layout(height=280)
    return fig


def run_fast_screening(patient_data: dict, trial_protocol: str, trial_id: str, progress_container=None) -> dict:
    """Run FAST screening with real-time progress updates."""
    try:
        from src.agents.supervisor_fast import FastSupervisorAgent
        agent = FastSupervisorAgent()

        progress_bar = None
        status_text = None
        if progress_container:
            progress_bar = progress_container.progress(0, text="Initializing...")
            status_text = progress_container.empty()

        current_progress = [0]

        def update_progress(message: str):
            if progress_bar:
                if "Step 1" in message:
                    current_progress[0] = 30
                elif "Step 2" in message:
                    current_progress[0] = 70
                progress_bar.progress(current_progress[0], text=message)
            if status_text:
                status_text.info(f"Processing: {message}")

        async def _async_screen():
            return await agent.screen_patient(
                patient_data=patient_data,
                trial_protocol=trial_protocol,
                trial_id=trial_id,
                progress_callback=update_progress
            )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_async_screen())
        finally:
            loop.close()

        if progress_bar:
            progress_bar.progress(100, text="Complete!")
        if status_text:
            status_text.success("Screening complete!")

        return result

    except Exception as e:
        st.error(f"Screening error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


# =============================================================================
# SIDEBAR - API STATUS & CONTROLS
# =============================================================================

api_key = get_api_key()

# API Status indicator (small green/red dot)
if api_key:
    st.sidebar.markdown(
        '<div class="api-status api-status-ok"><span class="green-dot"></span> API Ready</div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div class="api-status api-status-error"><span class="red-dot"></span> API not configured</div>',
        unsafe_allow_html=True
    )
    user_key = st.sidebar.text_input("Enter API Key", type="password", key="api_input")
    if user_key:
        st.session_state.google_api_key = user_key
        os.environ["GOOGLE_API_KEY"] = user_key
        st.rerun()

# Developer Login
st.sidebar.markdown("---")
if not is_developer_mode():
    with st.sidebar.expander("Developer Login"):
        dev_user = st.text_input("Username", key="dev_user")
        dev_pass = st.text_input("Password", type="password", key="dev_pass")
        if st.button("Login", key="dev_login"):
            if check_developer_login(dev_user, dev_pass):
                st.session_state.developer_logged_in = True
                st.success("Developer mode activated!")
                st.rerun()
            else:
                st.error("Invalid credentials")
else:
    st.sidebar.markdown('<span class="dev-badge">Developer Mode</span>', unsafe_allow_html=True)
    if st.sidebar.button("Logout Developer"):
        st.session_state.developer_logged_in = False
        st.rerun()

# Clear & New Patient button
st.sidebar.markdown("---")
if st.sidebar.button("Clear & New Patient", use_container_width=True):
    clear_session()
    st.rerun()

# Trial History
st.sidebar.markdown("---")
st.sidebar.subheader("Trial Selection")

trial_history = get_trial_history()
if trial_history:
    st.sidebar.caption("Recent Trials:")
    for i, trial in enumerate(trial_history[:8]):
        trial_id_hist = trial.get("trial_id", "Unknown")
        if st.sidebar.button(f"{trial_id_hist}", key=f"hist_{i}", use_container_width=True):
            st.session_state.selected_trial_id = trial_id_hist
            st.rerun()

# Trial input
default_trial = st.session_state.get("selected_trial_id", "")
trial_id = st.sidebar.text_input("Trial ID", value=default_trial, placeholder="NCT12345678")
trial_protocol = st.sidebar.text_area("Protocol (optional)", height=100, placeholder="Paste protocol or leave empty for default")


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("Clinical Trial Screening")
st.markdown("**Optimized AI System** for rapid patient-trial matching with explainable decisions.")

# Create tabs based on developer mode
if is_developer_mode():
    tab1, tab2, tab3, tab4 = st.tabs(["Patient Form", "Batch Processing", "JSON Input (Dev)", "Results"])
else:
    tab1, tab2, tab3 = st.tabs(["Patient Form", "Batch Processing", "Results"])


# =============================================================================
# TAB 1: PATIENT FORM
# =============================================================================

with tab1:
    st.header("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        patient_id = st.text_input("Patient ID", value="PT001")
        age = st.number_input("Age", min_value=0, max_value=120, value=55)
        sex = st.selectbox("Sex", ["male", "female", "other"])

        st.subheader("Diagnoses")
        diagnosis = st.text_input("Primary Condition", placeholder="Type 2 Diabetes Mellitus")
        icd10 = st.text_input("ICD-10 Code", placeholder="E11.9")

    with col2:
        st.subheader("Medications")
        medication = st.text_input("Medication", placeholder="Metformin")
        dose = st.text_input("Dose", placeholder="1000mg twice daily")

        st.subheader("Lab Values")
        lab_test = st.text_input("Test Name", placeholder="HbA1c")
        lab_value = st.number_input("Value", value=8.0, format="%.1f")
        lab_unit = st.text_input("Unit", placeholder="%")

    # Other Information field
    st.subheader("Other Information")
    other_info = st.text_area(
        "Additional Notes",
        placeholder="Enter any other relevant patient information, medical history, allergies, lifestyle factors, etc.",
        height=100
    )

    if st.button("Load Patient", type="primary", key="build"):
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "diagnoses": [{"condition": diagnosis, "icd10": icd10}] if diagnosis else [],
            "medications": [{"drug_name": medication, "dose": dose}] if medication else [],
            "lab_values": [{"test": lab_test, "value": lab_value, "unit": lab_unit}] if lab_test else [],
            "other_information": other_info if other_info else None
        }

        # Backend validation
        is_valid, msg = validate_patient_data(patient_data)
        if is_valid:
            st.session_state.patient_data = patient_data
            st.session_state.patient_validated = True
            st.success(f"Patient {patient_id} loaded successfully!")
        else:
            st.error(f"Validation error: {msg}")


# =============================================================================
# TAB 2: BATCH PROCESSING
# =============================================================================

with tab2:
    st.header("Batch Processing")
    st.markdown("Upload your patient files and clinical trial protocols for batch screening.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Files")
        patients_file = st.file_uploader(
            "Upload Patients (CSV or JSON)",
            type=["csv", "json"],
            help="CSV with columns: patient_id, age, sex, diagnoses, medications, lab_values. Or JSON array of patient objects."
        )

        if patients_file:
            try:
                if patients_file.name.endswith(".csv"):
                    patients_df = pd.read_csv(patients_file)
                    st.success(f"Loaded {len(patients_df)} patients from CSV")
                    st.dataframe(patients_df.head(), use_container_width=True)
                    patients_list = patients_df.to_dict('records')
                else:
                    patients_list = json.loads(patients_file.read().decode('utf-8'))
                    if isinstance(patients_list, dict):
                        patients_list = [patients_list]
                    st.success(f"Loaded {len(patients_list)} patients from JSON")

                st.session_state.batch_patients = patients_list
            except Exception as e:
                st.error(f"Error loading patients file: {e}")

    with col2:
        st.subheader("Clinical Trial Protocols")
        protocols_file = st.file_uploader(
            "Upload Protocols (CSV, JSON, or TXT)",
            type=["csv", "json", "txt", "md"],
            help="CSV with columns: trial_id, protocol_text. Or JSON with trial_id keys. Or single protocol text file."
        )

        if protocols_file:
            try:
                content = protocols_file.read().decode('utf-8')
                if protocols_file.name.endswith(".csv"):
                    import io
                    protocols_df = pd.read_csv(io.StringIO(content))
                    protocols_dict = dict(zip(protocols_df['trial_id'], protocols_df['protocol_text']))
                    st.success(f"Loaded {len(protocols_dict)} protocols from CSV")
                elif protocols_file.name.endswith(".json"):
                    protocols_dict = json.loads(content)
                    st.success(f"Loaded {len(protocols_dict)} protocols from JSON")
                else:
                    protocols_dict = {"UPLOADED": content}
                    st.success("Loaded protocol text file")

                st.session_state.uploaded_protocols = protocols_dict
            except Exception as e:
                st.error(f"Error loading protocols file: {e}")

    st.markdown("---")

    batch_trial_id = st.text_input("Default Trial ID for Batch", value="NCT12345678", key="batch_trial")

    col1, col2, col3 = st.columns(3)

    with col1:
        run_batch = st.button("Run Batch Screening", type="primary", use_container_width=True)

    with col2:
        if st.button("Clear Batch Results", use_container_width=True):
            st.session_state.batch_results = []
            st.rerun()

    with col3:
        if st.session_state.batch_results:
            df_results = pd.DataFrame(st.session_state.batch_results)
            csv = df_results.to_csv(index=False)
            st.download_button(
                "Export Results CSV",
                csv,
                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

    if run_batch:
        patients_to_process = st.session_state.get("batch_patients", [])
        protocols = st.session_state.get("uploaded_protocols", {})

        if not patients_to_process:
            st.error("Please upload a patients file first")
        elif not api_key:
            st.error("API Key required for screening")
        else:
            st.session_state.batch_results = []
            progress_bar = st.progress(0)
            status = st.empty()

            for i, patient in enumerate(patients_to_process):
                is_valid, msg = validate_patient_data(patient)
                if not is_valid:
                    st.session_state.batch_results.append({
                        "patient_id": patient.get("patient_id", f"Patient_{i}"),
                        "trial_id": batch_trial_id,
                        "decision": "ERROR",
                        "confidence": 0,
                        "error": msg
                    })
                    continue

                status.info(f"Processing {patient.get('patient_id', f'Patient_{i}')} ({i+1}/{len(patients_to_process)})")
                progress_bar.progress((i + 1) / len(patients_to_process))

                protocol = protocols.get(batch_trial_id) or protocols.get("UPLOADED") or f"""
                CLINICAL TRIAL: {batch_trial_id}
                INCLUSION: Age 18-75, Type 2 Diabetes, HbA1c 7-10%
                EXCLUSION: Type 1 Diabetes, Pregnancy, Severe renal impairment
                """

                try:
                    result = run_fast_screening(patient, protocol, batch_trial_id)
                    st.session_state.batch_results.append({
                        "patient_id": patient.get("patient_id"),
                        "trial_id": batch_trial_id,
                        "decision": result.get("decision", "UNKNOWN") if result else "ERROR",
                        "confidence": result.get("confidence", 0) if result else 0,
                        "narrative": (result.get("clinical_narrative", "")[:100] + "...") if result else ""
                    })
                except Exception as e:
                    st.session_state.batch_results.append({
                        "patient_id": patient.get("patient_id"),
                        "trial_id": batch_trial_id,
                        "decision": "ERROR",
                        "confidence": 0,
                        "error": str(e)
                    })

            progress_bar.progress(100)
            status.success(f"Batch complete! {len(patients_to_process)} patients processed")

    if st.session_state.batch_results:
        st.subheader("Batch Results")
        df = pd.DataFrame(st.session_state.batch_results)
        st.dataframe(df, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        decisions = [r.get("decision") for r in st.session_state.batch_results]
        with col1:
            st.metric("Eligible", decisions.count("ELIGIBLE"))
        with col2:
            st.metric("Ineligible", decisions.count("INELIGIBLE"))
        with col3:
            st.metric("Uncertain", decisions.count("UNCERTAIN"))
        with col4:
            st.metric("Errors", decisions.count("ERROR"))


# =============================================================================
# TAB 3 (DEV ONLY): JSON INPUT
# =============================================================================

if is_developer_mode():
    with tab3:
        st.header("Developer JSON Input")
        st.warning("This tab is only visible in Developer Mode")

        json_template = """{
    "patient_id": "PT001",
    "age": 58,
    "sex": "male",
    "diagnoses": [{"condition": "Type 2 Diabetes Mellitus", "icd10": "E11.9"}],
    "medications": [{"drug_name": "Metformin", "dose": "1000mg twice daily"}],
    "lab_values": [{"test": "HbA1c", "value": 8.2, "unit": "%"}],
    "other_information": "No known allergies"
}"""

        json_input = st.text_area("Patient JSON", value=json_template, height=300)

        if st.button("Parse & Load JSON", key="parse_json"):
            try:
                parsed_data = json.loads(json_input)
                is_valid, msg = validate_patient_data(parsed_data)
                if is_valid:
                    st.session_state.patient_data = parsed_data
                    st.session_state.patient_validated = True
                    st.success(f"Patient {parsed_data.get('patient_id', 'N/A')} loaded!")
                    st.json(parsed_data)
                else:
                    st.error(f"Validation error: {msg}")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

        if st.session_state.patient_data:
            st.subheader("Current Patient Data (JSON)")
            st.json(st.session_state.patient_data)


# =============================================================================
# RESULTS TAB
# =============================================================================

results_tab = tab4 if is_developer_mode() else tab3

with results_tab:
    st.header("Run Screening & Results")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.patient_data:
            st.success(f"Patient loaded: {st.session_state.patient_data.get('patient_id', 'N/A')}")
        else:
            st.warning("No patient loaded")
    with col2:
        if trial_id:
            st.success(f"Trial: {trial_id}")
        else:
            st.warning("No trial selected")

    progress_container = st.container()

    if st.button("Run Eligibility Screening", type="primary", use_container_width=True):
        if not api_key:
            st.error("API Key required")
        elif not st.session_state.patient_data:
            st.error("Please load patient data first")
        elif not trial_id:
            st.error("Please enter a trial ID")
        else:
            start_time = time.time()

            protocol = trial_protocol if trial_protocol else f"""
            CLINICAL TRIAL: {trial_id}
            INCLUSION CRITERIA:
            1. Age 18-75 years
            2. Diagnosis of Type 2 Diabetes Mellitus
            3. HbA1c between 7.0% and 10.0%
            4. Currently on stable metformin therapy
            EXCLUSION CRITERIA:
            1. Type 1 Diabetes
            2. Pregnant or nursing women
            3. Severe renal impairment (eGFR < 30 mL/min)
            """

            result = run_fast_screening(
                st.session_state.patient_data,
                protocol,
                trial_id,
                progress_container
            )

            elapsed = time.time() - start_time

            if result:
                result["elapsed_time"] = f"{elapsed:.1f}s"
                st.session_state.screening_result = result
                save_trial_to_db(trial_id, protocol, result)

    if st.session_state.screening_result:
        result = st.session_state.screening_result
        st.divider()

        decision = result.get("decision", "UNCERTAIN")
        confidence = result.get("confidence", 0.0)
        elapsed = result.get("elapsed_time", "N/A")

        st.markdown(
            f'<div class="{get_eligibility_class(decision)}">{decision}</div>',
            unsafe_allow_html=True
        )

        st.caption(f"Completed in {elapsed}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Confidence", f"{confidence:.0%}")
        with col2:
            st.metric("Level", result.get("confidence_level", "N/A"))
        with col3:
            review = "Yes" if result.get("requires_human_review") else "No"
            st.metric("Human Review", review)
        with col4:
            st.metric("Trial", trial_id)

        res_tab1, res_tab2, res_tab3 = st.tabs(["Analysis", "Explainability", "Narrative"])

        with res_tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = create_confidence_gauge(confidence)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Key Factors:**")
                for factor in result.get("key_factors", []):
                    st.markdown(f"- {factor}")
                if result.get("concerns"):
                    st.markdown("**Concerns:**")
                    for concern in result.get("concerns", []):
                        st.markdown(f"- {concern}")

        with res_tab2:
            data = result.get("explainability_table", [])
            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.info("No detailed explainability data available")

        with res_tab3:
            narrative = result.get("clinical_narrative", "")
            if narrative:
                st.markdown(narrative)
            else:
                st.info("No narrative generated")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Export Results JSON",
                json.dumps(result, indent=2),
                f"screening_{trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        with col2:
            if is_developer_mode():
                with st.expander("Raw JSON (Dev)"):
                    st.json(result)


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown("""
**Clinical Trial Screening** | v3.0.0 | Powered by AI

2026 CodeNoLimits - Melea & David
""")

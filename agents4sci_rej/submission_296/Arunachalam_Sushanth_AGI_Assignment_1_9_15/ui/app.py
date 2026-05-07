"""
Streamlit UI Dashboard for IDP Admissions System
Main application with tabs for upload, dashboard, detail view, chat bot, and settings
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import yaml

# Add code directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "code"))

from service_client import ServiceClient
from components import (
    render_score_badge, 
    render_evidence_viewer, 
    render_risk_banner,
    render_applicant_summary
)
from bot import ChatBot

# Configure Streamlit page
st.set_page_config(
    page_title="IDP Admissions System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'service_client' not in st.session_state:
    st.session_state.service_client = ServiceClient()
if 'chat_bot' not in st.session_state:
    st.session_state.chat_bot = ChatBot()

def main():
    """Main application interface"""
    
    # Sidebar configuration
    with st.sidebar:
        st.title("🎓 IDP Admissions")
        st.markdown("---")
        
        # System status
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processed", get_processed_count())
        with col2:
            st.metric("Pending", get_pending_count())
        
        # Quick stats
        if st.button("🔄 Refresh Stats"):
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Navigation**")
        st.markdown("- 📤 **Upload**: Process new applications")
        st.markdown("- 📊 **Dashboard**: View all applications")
        st.markdown("- 👤 **Detail**: Individual application view")
        st.markdown("- 🤖 **Chat**: Ask questions about results")
        st.markdown("- ⚙️ **Settings**: Configure thresholds")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload", 
        "📊 Dashboard", 
        "👤 Applicant Detail", 
        "🤖 Chat Bot", 
        "⚙️ Settings"
    ])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_dashboard_tab()
    
    with tab3:
        render_detail_tab()
    
    with tab4:
        render_chat_tab()
    
    with tab5:
        render_settings_tab()

def render_upload_tab():
    """Upload and processing interface"""
    st.header("📤 Document Upload & Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Application Documents")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Choose PDF files (transcripts, resumes, statements)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload academic transcripts, resumes, and statements of purpose"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} file(s) selected")
            
            # Processing options
            col_a, col_b = st.columns(2)
            with col_a:
                ocr_backend = st.selectbox(
                    "OCR Backend",
                    ["auto", "pdfminer", "simulated"],
                    help="Choose OCR backend for text extraction"
                )
            
            with col_b:
                program = st.selectbox(
                    "Target Program",
                    ["general", "computer_science", "engineering", "business"],
                    help="Select target academic program for evaluation"
                )
            
            # Process button
            if st.button("🚀 Process Applications", type="primary"):
                process_uploaded_files(uploaded_files, ocr_backend, program)
    
    with col2:
        st.subheader("📋 Processing Guidelines")
        
        st.info("""
        **Document Requirements:**
        - **Transcripts**: PDF format, clear text
        - **Resumes**: Standard format with sections
        - **Statements**: Text-readable PDF files
        
        **Processing Steps:**
        1. OCR text extraction
        2. Document parsing & analysis
        3. Academic decision generation
        4. Evidence grounding
        5. Results storage
        """)
        
        # Recent uploads
        st.subheader("📈 Recent Activity")
        recent_files = get_recent_uploads()
        if recent_files:
            for file_info in recent_files[:3]:
                st.write(f"✅ {file_info['name']} - {file_info['status']}")
        else:
            st.write("No recent uploads")

def render_dashboard_tab():
    """Applications dashboard with filtering and export"""
    st.header("📊 Applications Dashboard")
    
    # Load processed applications
    applications = load_processed_applications()
    
    if not applications:
        st.warning("No processed applications found. Upload some documents first!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(applications)
    
    # Filters section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        decision_filter = st.multiselect(
            "Filter by Decision",
            options=df['decision'].unique(),
            default=df['decision'].unique()
        )
    
    with col2:
        confidence_range = st.slider(
            "Confidence Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.1
        )
    
    with col3:
        program_filter = st.multiselect(
            "Filter by Program",
            options=df.get('program', pd.Series(['general'])).unique(),
            default=df.get('program', pd.Series(['general'])).unique()
        )
    
    with col4:
        show_warnings = st.checkbox("Show Only Applications with Warnings", value=False)
    
    # Apply filters
    filtered_df = df[
        (df['decision'].isin(decision_filter)) &
        (df['confidence'].between(confidence_range[0], confidence_range[1]))
    ]
    
    if show_warnings and 'warnings' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['warnings'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", len(filtered_df))
    with col2:
        accept_rate = len(filtered_df[filtered_df['decision'] == 'ACCEPT_ACADEMIC']) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric("Accept Rate", f"{accept_rate:.1f}%")
    with col3:
        avg_confidence = filtered_df['confidence'].mean() if len(filtered_df) > 0 else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    with col4:
        abstention_rate = len(filtered_df[filtered_df['decision'] == 'ABSTAIN']) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric("Abstention Rate", f"{abstention_rate:.1f}%")
    
    # Applications table
    st.subheader("Applications Overview")
    
    # Format data for display
    display_df = filtered_df.copy()
    display_df['Decision'] = display_df['decision'].apply(lambda x: render_score_badge(x, 'decision'))
    display_df['GPA'] = display_df['gpa'].apply(lambda x: f"{x:.2f}")
    display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2f}")
    display_df['Credits'] = display_df['credits'].apply(lambda x: f"{x:.0f}")
    
    # Select columns for display
    display_columns = ['application_id', 'Decision', 'GPA', 'Credits', 'Confidence', 'timestamp']
    if 'program' in display_df.columns:
        display_columns.insert(-1, 'program')
    
    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Export options
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📥 Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"applications_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def render_detail_tab():
    """Individual applicant detail view"""
    st.header("👤 Applicant Detail View")
    
    # Load applications for selection
    applications = load_processed_applications()
    
    if not applications:
        st.warning("No processed applications available.")
        return
    
    # Application selector
    app_ids = [app['application_id'] for app in applications]
    selected_app_id = st.selectbox("Select Application", app_ids)
    
    if selected_app_id:
        # Find selected application
        app_data = next((app for app in applications if app['application_id'] == selected_app_id), None)
        
        if app_data:
            render_applicant_detail(app_data)

def render_applicant_detail(app_data):
    """Render detailed view for a specific applicant"""
    
    # Summary section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("GPA", f"{app_data['gpa']:.2f}")
        st.metric("Credits", f"{app_data['credits']:.0f}")
    
    with col2:
        decision_color = {
            'ACCEPT_ACADEMIC': 'green',
            'REVIEW': 'orange', 
            'REJECT_ACADEMIC': 'red',
            'ABSTAIN': 'gray'
        }
        st.markdown(f"**Decision:** :{decision_color.get(app_data['decision'], 'gray')}[{app_data['decision']}]")
        st.metric("Confidence", f"{app_data['confidence']:.2f}")
    
    with col3:
        st.write(f"**Processed:** {app_data['timestamp']}")
        if app_data.get('warnings'):
            st.warning(f"⚠️ {len(app_data['warnings'])} warnings")
    
    # Risk banner
    if app_data['confidence'] < 0.7 or app_data['decision'] == 'ABSTAIN':
        render_risk_banner("Low confidence decision - human review recommended")
    
    # Tabbed detail view
    detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
        "📋 Summary", "📄 Evidence", "🔍 Consistency", "⚡ Actions"
    ])
    
    with detail_tab1:
        render_applicant_summary(app_data)
    
    with detail_tab2:
        render_evidence_viewer(app_data.get('evidence', {}))
    
    with detail_tab3:
        render_consistency_check(app_data)
    
    with detail_tab4:
        render_action_buttons(app_data)

def render_consistency_check(app_data):
    """Render consistency check results"""
    st.subheader("🔍 Cross-Document Consistency")
    
    # Mock consistency checks
    checks = [
        {"check": "GPA Recomputation", "status": "✅ Pass", "details": "Computed GPA matches extracted value"},
        {"check": "Credit Hour Validation", "status": "✅ Pass", "details": "Credit hours within reasonable range"},
        {"check": "Date Consistency", "status": "⚠️ Warning", "details": "Some graduation dates may be inconsistent"},
        {"check": "Program Alignment", "status": "✅ Pass", "details": "Prerequisites match target program"}
    ]
    
    for check in checks:
        col1, col2, col3 = st.columns([2, 1, 3])
        with col1:
            st.write(check["check"])
        with col2:
            st.write(check["status"])
        with col3:
            st.write(check["details"])

def render_action_buttons(app_data):
    """Render action buttons for application"""
    st.subheader("⚡ Available Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👥 Escalate to Human"):
            st.success("Application escalated for human review")
    
    with col2:
        if st.button("🔄 Reprocess"):
            st.info("Application queued for reprocessing")
    
    with col3:
        if st.button("📊 View Analytics"):
            st.info("Analytics view not implemented")
    
    # Escalation form
    if app_data['decision'] == 'ABSTAIN' or app_data['confidence'] < 0.7:
        st.subheader("📝 Escalation Notes")
        escalation_note = st.text_area("Add notes for human reviewer:")
        if st.button("Submit Escalation"):
            if escalation_note:
                st.success("Escalation submitted with notes")
            else:
                st.error("Please add escalation notes")

def render_chat_tab():
    """Chat bot interface"""
    st.header("🤖 IDP Assistant Chat Bot")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat interface
        st.subheader("Ask questions about applications and results")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Assistant:** {message['content']}")
        
        # Input area
        user_input = st.text_input("Ask a question:", placeholder="e.g., What's the average GPA of accepted applicants?")
        
        if st.button("Send") and user_input:
            # Add user message to history
            st.session_state.chat_history.append({'type': 'user', 'content': user_input})
            
            # Get bot response
            response = st.session_state.chat_bot.get_response(user_input)
            st.session_state.chat_history.append({'type': 'bot', 'content': response})
            
            st.rerun()
        
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What's the average GPA of applicants?",
            "How many applications were abstained?",
            "Show me the confidence distribution",
            "What are common warning types?",
            "Which program has highest accept rate?"
        ]
        
        for question in sample_questions:
            if st.button(f"📌 {question}", key=f"sample_{question[:10]}"):
                st.session_state.chat_history.append({'type': 'user', 'content': question})
                response = st.session_state.chat_bot.get_response(question)
                st.session_state.chat_history.append({'type': 'bot', 'content': response})
                st.rerun()

def render_settings_tab():
    """Settings and configuration interface"""
    st.header("⚙️ System Settings")
    
    # Load current config
    config = load_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Decision Thresholds")
        
        # Global thresholds
        gpa_threshold = st.slider(
            "GPA Threshold",
            min_value=2.0,
            max_value=4.0,
            value=config.get("thresholds", {}).get("gpa_threshold", 3.0),
            step=0.1,
            help="Minimum GPA for academic acceptance"
        )
        
        min_credits = st.slider(
            "Minimum Credits",
            min_value=60,
            max_value=150,
            value=config.get("thresholds", {}).get("min_credits", 90),
            step=5,
            help="Minimum credit hours required"
        )
        
        abstain_threshold = st.slider(
            "Abstention Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.get("thresholds", {}).get("abstain_threshold", 0.7),
            step=0.05,
            help="Confidence threshold below which system abstains"
        )
    
    with col2:
        st.subheader("🏫 Program-Specific Rules")
        
        programs = ["computer_science", "engineering", "business"]
        selected_program = st.selectbox("Select Program", programs)
        
        program_rules = config.get("thresholds", {}).get("program_rules", {}).get(selected_program, {})
        
        prog_gpa = st.slider(
            f"GPA Threshold ({selected_program})",
            min_value=2.0,
            max_value=4.0,
            value=program_rules.get("gpa_threshold", gpa_threshold),
            step=0.1
        )
        
        prog_credits = st.slider(
            f"Min Credits ({selected_program})",
            min_value=60,
            max_value=150,
            value=program_rules.get("min_credits", min_credits),
            step=5
        )
    
    # Preview section
    st.subheader("🔍 Settings Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Preview Changes"):
            st.info("Settings preview:")
            st.json({
                "global": {
                    "gpa_threshold": gpa_threshold,
                    "min_credits": min_credits,
                    "abstain_threshold": abstain_threshold
                },
                selected_program: {
                    "gpa_threshold": prog_gpa,
                    "min_credits": prog_credits
                }
            })
    
    with col2:
        if st.button("💾 Save Settings", type="primary"):
            # Update config
            updated_config = config.copy()
            updated_config["thresholds"]["gpa_threshold"] = gpa_threshold
            updated_config["thresholds"]["min_credits"] = min_credits
            updated_config["thresholds"]["abstain_threshold"] = abstain_threshold
            
            if "program_rules" not in updated_config["thresholds"]:
                updated_config["thresholds"]["program_rules"] = {}
            
            updated_config["thresholds"]["program_rules"][selected_program] = {
                "gpa_threshold": prog_gpa,
                "min_credits": prog_credits
            }
            
            # Save config
            save_config(updated_config)
            st.success("✅ Settings saved successfully!")

# Utility functions

def get_processed_count():
    """Get count of processed applications"""
    applications = load_processed_applications()
    return len(applications)

def get_pending_count():
    """Get count of pending applications"""
    incoming_dir = Path("incoming")
    if incoming_dir.exists():
        return len(list(incoming_dir.glob("*.pdf")))
    return 0

def get_recent_uploads():
    """Get recent upload information"""
    # Mock data for demonstration
    return [
        {"name": "transcript_001.pdf", "status": "Processed"},
        {"name": "resume_002.pdf", "status": "Processed"},
        {"name": "statement_003.pdf", "status": "Processing"},
    ]

def process_uploaded_files(uploaded_files, ocr_backend, program):
    """Process uploaded files"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Save file temporarily
        temp_path = Path("incoming") / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process using service client
        try:
            result = st.session_state.service_client.process_file(str(temp_path), ocr_backend, program)
            results.append(result)
            status = "✅ Success"
        except Exception as e:
            status = f"❌ Error: {str(e)}"
            results.append({"error": str(e), "filename": uploaded_file.name})
        
        progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text(f"Processed {uploaded_file.name}: {status}")
    
    # Show results summary
    st.success(f"Processed {len(uploaded_files)} files")
    
    # Display results table
    if results:
        results_df = pd.DataFrame([
            {
                "File": r.get("filename", "unknown"),
                "Decision": r.get("decision", "Error"),
                "GPA": r.get("gpa", "N/A"),
                "Confidence": r.get("confidence", "N/A")
            }
            for r in results
        ])
        st.dataframe(results_df, use_container_width=True)

def load_processed_applications():
    """Load processed applications from JSON files"""
    processed_dir = Path("processed")
    applications = []
    
    if processed_dir.exists():
        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    app_data = json.load(f)
                    applications.append(app_data)
            except Exception as e:
                st.error(f"Error loading {json_file}: {e}")
    
    return applications

def load_config():
    """Load system configuration"""
    config_path = Path("config/config.yaml")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {"thresholds": {"gpa_threshold": 3.0, "min_credits": 90, "abstain_threshold": 0.7}}

def save_config(config):
    """Save system configuration"""
    config_path = Path("config/config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    main()
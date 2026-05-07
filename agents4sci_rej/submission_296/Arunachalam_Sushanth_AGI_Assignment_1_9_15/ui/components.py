"""
Reusable UI components for the Streamlit dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

def render_score_badge(score: Any, score_type: str) -> str:
    """Render a colored badge for scores or decisions"""
    
    if score_type == "decision":
        colors = {
            "ACCEPT_ACADEMIC": "🟢",
            "REVIEW": "🟡", 
            "REJECT_ACADEMIC": "🔴",
            "ABSTAIN": "⚪"
        }
        return f"{colors.get(score, '❓')} {score}"
    
    elif score_type == "confidence":
        if score >= 0.8:
            return f"🟢 {score:.2f}"
        elif score >= 0.6:
            return f"🟡 {score:.2f}"
        else:
            return f"🔴 {score:.2f}"
    
    elif score_type == "gpa":
        if score >= 3.5:
            return f"🟢 {score:.2f}"
        elif score >= 3.0:
            return f"🟡 {score:.2f}"
        else:
            return f"🔴 {score:.2f}"
    
    return str(score)

def render_evidence_viewer(evidence: Dict[str, Any]):
    """Render evidence spans in an organized view"""
    
    if not evidence:
        st.info("No evidence data available")
        return
    
    # Transcript evidence
    transcript_spans = evidence.get("transcript_spans", [])
    if transcript_spans:
        st.subheader("📄 Transcript Evidence")
        
        for i, span in enumerate(transcript_spans):
            with st.expander(f"Evidence {i+1}: {span.get('type', 'unknown').title()}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Text:** {span.get('text', 'N/A')}")
                    if 'metadata' in span:
                        metadata = span['metadata']
                        st.write(f"**Credits:** {metadata.get('credits', 'N/A')}")
                        st.write(f"**Grade:** {metadata.get('grade', 'N/A')}")
                        st.write(f"**Grade Points:** {metadata.get('grade_points', 'N/A')}")
                
                with col2:
                    st.write(f"**Position:** {span.get('start', 0)}-{span.get('end', 0)}")
                    st.write(f"**Type:** {span.get('type', 'unknown')}")
    
    # Resume entities
    resume_entities = evidence.get("resume_entities", [])
    if resume_entities:
        st.subheader("📋 Resume Entities")
        
        # Group by entity type
        entities_by_type = {}
        for entity in resume_entities:
            entity_type = entity.get('type', 'unknown')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        for entity_type, entities in entities_by_type.items():
            with st.expander(f"{entity_type} ({len(entities)} found)"):
                for entity in entities:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"• {entity.get('text', 'N/A')}")
                    with col2:
                        confidence = entity.get('confidence', 0)
                        st.write(render_score_badge(confidence, "confidence"))
    
    # Statement of purpose spans
    sop_spans = evidence.get("sop_spans", [])
    if sop_spans:
        st.subheader("✍️ Statement Evidence")
        
        for span in sop_spans:
            rubric_dim = span.get('rubric_dimension', 'unknown')
            score = span.get('score', 0)
            
            with st.expander(f"{rubric_dim.replace('_', ' ').title()} (Score: {score}/5)"):
                st.write(f"**Evidence Text:**")
                st.write(span.get('text', 'N/A'))
                
                progress_value = score / 5.0
                st.progress(progress_value)

def render_risk_banner(message: str, risk_level: str = "warning"):
    """Render a risk assessment banner"""
    
    if risk_level == "high":
        st.error(f"🚨 **HIGH RISK:** {message}")
    elif risk_level == "medium":
        st.warning(f"⚠️ **MEDIUM RISK:** {message}")
    else:
        st.info(f"ℹ️ **NOTICE:** {message}")

def render_applicant_summary(app_data: Dict[str, Any]):
    """Render comprehensive applicant summary"""
    
    st.subheader("📊 Application Summary")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="GPA", 
            value=f"{app_data.get('gpa', 0):.2f}",
            delta=f"{app_data.get('gpa', 0) - 3.0:.2f} vs 3.0 threshold"
        )
    
    with col2:
        st.metric(
            label="Credits", 
            value=f"{app_data.get('credits', 0):.0f}",
            delta=f"{app_data.get('credits', 0) - 90:.0f} vs 90 minimum"
        )
    
    with col3:
        confidence = app_data.get('confidence', 0)
        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}",
            delta=f"{confidence - 0.7:.2f} vs 0.7 threshold"
        )
    
    with col4:
        processing_time = app_data.get('processing_info', {}).get('processing_time_seconds', 0)
        st.metric(
            label="Process Time",
            value=f"{processing_time:.1f}s"
        )
    
    # Decision reasoning
    st.subheader("🎯 Decision Analysis")
    
    decision = app_data.get('decision')
    reason = app_data.get('reason', 'No reason provided')
    
    decision_color = {
        'ACCEPT_ACADEMIC': 'success',
        'REVIEW': 'warning',
        'REJECT_ACADEMIC': 'error',
        'ABSTAIN': 'info'
    }.get(decision, 'info')
    
    if decision_color == 'success':
        st.success(f"**Decision:** {decision}")
    elif decision_color == 'warning':
        st.warning(f"**Decision:** {decision}")
    elif decision_color == 'error':
        st.error(f"**Decision:** {decision}")
    else:
        st.info(f"**Decision:** {decision}")
    
    st.write(f"**Reasoning:** {reason}")
    
    # Warnings section
    warnings = app_data.get('warnings', [])
    if warnings:
        st.subheader("⚠️ Processing Warnings")
        for warning in warnings:
            st.warning(f"• {warning}")
    
    # Additional details if available
    details = app_data.get('details', {})
    if details:
        st.subheader("📋 Additional Details")
        
        with st.expander("Show Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Program:** {details.get('program', 'N/A')}")
                st.write(f"**GPA Threshold:** {details.get('gpa_threshold', 'N/A')}")
                st.write(f"**Min Credits:** {details.get('min_credits', 'N/A')}")
            
            with col2:
                st.write(f"**Parsing Confidence:** {details.get('parsing_confidence', 'N/A')}")
                st.write(f"**Primary Decision:** {details.get('primary_decision', 'N/A')}")
    
    # Processing information
    processing_info = app_data.get('processing_info', {})
    if processing_info:
        st.subheader("🔧 Processing Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**OCR Backend:** {processing_info.get('ocr_backend', 'N/A')}")
            st.write(f"**Processing Time:** {processing_info.get('processing_time_seconds', 0):.2f}s")
        
        with col2:
            docs_processed = processing_info.get('documents_processed', [])
            st.write(f"**Documents Processed:** {len(docs_processed)}")
            
            if docs_processed:
                for doc in docs_processed:
                    doc_type = doc.get('document_type', 'unknown')
                    pages = doc.get('pages', 1)
                    st.write(f"  • {doc_type}: {pages} page(s)")

def render_confidence_gauge(confidence: float):
    """Render a gauge chart for confidence score"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence"},
        delta = {'reference': 0.7},
        gauge = {
            'axis': {'range': [None, 1.0]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.5], 'color': "lightgray"},
                {'range': [0.5, 0.7], 'color': "yellow"},
                {'range': [0.7, 1.0], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def render_gpa_distribution_chart(applications: List[Dict[str, Any]]):
    """Render GPA distribution histogram"""
    
    gpas = [app.get('gpa', 0) for app in applications]
    
    fig = px.histogram(
        x=gpas,
        nbins=20,
        title="GPA Distribution",
        labels={'x': 'GPA', 'y': 'Count'},
        color_discrete_sequence=['lightblue']
    )
    
    fig.add_vline(x=3.0, line_dash="dash", line_color="red", 
                  annotation_text="Threshold: 3.0")
    
    fig.update_layout(height=400)
    return fig

def render_decision_pie_chart(applications: List[Dict[str, Any]]):
    """Render decision distribution pie chart"""
    
    decisions = [app.get('decision', 'Unknown') for app in applications]
    decision_counts = {}
    for decision in decisions:
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    fig = px.pie(
        values=list(decision_counts.values()),
        names=list(decision_counts.keys()),
        title="Decision Distribution",
        color_discrete_map={
            'ACCEPT_ACADEMIC': 'green',
            'REVIEW': 'orange',
            'REJECT_ACADEMIC': 'red',
            'ABSTAIN': 'gray'
        }
    )
    
    fig.update_layout(height=400)
    return fig

def render_processing_time_chart(applications: List[Dict[str, Any]]):
    """Render processing time distribution"""
    
    processing_times = []
    for app in applications:
        time = app.get('processing_info', {}).get('processing_time_seconds', 0)
        if time > 0:
            processing_times.append(time)
    
    if not processing_times:
        st.info("No processing time data available")
        return None
    
    fig = px.histogram(
        x=processing_times,
        nbins=15,
        title="Processing Time Distribution",
        labels={'x': 'Processing Time (seconds)', 'y': 'Count'},
        color_discrete_sequence=['lightcoral']
    )
    
    avg_time = sum(processing_times) / len(processing_times)
    fig.add_vline(x=avg_time, line_dash="dash", line_color="blue",
                  annotation_text=f"Average: {avg_time:.1f}s")
    
    fig.update_layout(height=400)
    return fig
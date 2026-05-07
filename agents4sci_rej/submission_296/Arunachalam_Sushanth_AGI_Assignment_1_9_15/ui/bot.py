"""
Rule-based chat bot for answering questions about application data
No external LLM dependencies - uses template matching and data analysis
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import statistics

logger = logging.getLogger(__name__)

class ChatBot:
    """Rule-based chat bot for IDP system queries"""
    
    def __init__(self):
        self.applications_cache = None
        self.cache_timestamp = None
        
        # Initialize response templates
        self._initialize_templates()
        
        # Load applications data
        self._refresh_data()
    
    def _initialize_templates(self):
        """Initialize response templates for common queries"""
        
        self.templates = {
            "greeting": {
                "patterns": [r"hello", r"hi", r"hey", r"good morning", r"good afternoon"],
                "responses": [
                    "Hello! I'm here to help you analyze application data. What would you like to know?",
                    "Hi there! Ask me about GPAs, decisions, confidence scores, or any other application metrics.",
                    "Hey! I can help you understand the processed applications. What are you curious about?"
                ]
            },
            
            "average_gpa": {
                "patterns": [r"average gpa", r"mean gpa", r"avg gpa", r"typical gpa"],
                "handler": self._handle_average_gpa
            },
            
            "decision_stats": {
                "patterns": [r"decision", r"accept", r"reject", r"review", r"abstain"],
                "handler": self._handle_decision_stats
            },
            
            "confidence": {
                "patterns": [r"confidence", r"certainty", r"sure"],
                "handler": self._handle_confidence_stats
            },
            
            "warnings": {
                "patterns": [r"warning", r"problem", r"issue", r"error"],
                "handler": self._handle_warning_stats
            },
            
            "program": {
                "patterns": [r"program", r"major", r"department"],
                "handler": self._handle_program_stats
            },
            
            "processing_time": {
                "patterns": [r"processing time", r"how long", r"speed", r"fast"],
                "handler": self._handle_processing_time
            },
            
            "help": {
                "patterns": [r"help", r"what can you do", r"commands"],
                "responses": [
                    """I can help you analyze application data! Try asking:
                    
• "What's the average GPA?"
• "How many applications were accepted?"
• "Show me confidence distribution"
• "What are common warnings?"
• "Which program has the highest accept rate?"
• "How long does processing take?"
• "How many applications need human review?"

Just ask naturally and I'll do my best to help!"""
                ]
            }
        }
    
    def get_response(self, user_input: str) -> str:
        """Generate response to user input"""
        
        if not user_input.strip():
            return "Please ask me a question about the application data!"
        
        user_input_lower = user_input.lower()
        
        # Refresh data periodically
        self._refresh_data()
        
        # Check if we have data to analyze
        if not self.applications_cache:
            return "No application data available yet. Please process some applications first!"
        
        # Find matching template
        for template_name, template in self.templates.items():
            patterns = template.get("patterns", [])
            
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    
                    if "handler" in template:
                        try:
                            return template["handler"](user_input_lower)
                        except Exception as e:
                            logger.error(f"Handler error for {template_name}: {e}")
                            return "Sorry, I encountered an error analyzing that data."
                    
                    elif "responses" in template:
                        import random
                        return random.choice(template["responses"])
        
        # Default response for unmatched queries
        return self._handle_unknown_query(user_input)
    
    def _refresh_data(self):
        """Refresh applications data from processed files"""
        
        processed_dir = Path("processed")
        if not processed_dir.exists():
            return
        
        # Check if we need to refresh
        current_files = list(processed_dir.glob("*.json"))
        if self.cache_timestamp and current_files:
            latest_modification = max(f.stat().st_mtime for f in current_files)
            if latest_modification <= self.cache_timestamp:
                return  # No need to refresh
        
        # Load applications
        applications = []
        for json_file in current_files:
            try:
                with open(json_file, 'r') as f:
                    app_data = json.load(f)
                    applications.append(app_data)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        self.applications_cache = applications
        self.cache_timestamp = max(f.stat().st_mtime for f in current_files) if current_files else None
        
        logger.info(f"Refreshed data cache with {len(applications)} applications")
    
    def _handle_average_gpa(self, user_input: str) -> str:
        """Handle GPA-related queries"""
        
        gpas = [app.get('gpa', 0) for app in self.applications_cache if app.get('gpa', 0) > 0]
        
        if not gpas:
            return "No GPA data available in processed applications."
        
        avg_gpa = statistics.mean(gpas)
        median_gpa = statistics.median(gpas)
        min_gpa = min(gpas)
        max_gpa = max(gpas)
        
        # Determine which programs have highest/lowest GPAs
        gpa_by_program = {}
        for app in self.applications_cache:
            program = app.get('program', 'general')
            gpa = app.get('gpa', 0)
            if gpa > 0:
                if program not in gpa_by_program:
                    gpa_by_program[program] = []
                gpa_by_program[program].append(gpa)
        
        program_stats = {}
        for program, program_gpas in gpa_by_program.items():
            program_stats[program] = statistics.mean(program_gpas)
        
        response = f"""📊 **GPA Analysis** ({len(gpas)} applications):

• **Average GPA:** {avg_gpa:.2f}
• **Median GPA:** {median_gpa:.2f}
• **Range:** {min_gpa:.2f} - {max_gpa:.2f}

"""
        
        if program_stats:
            response += "**By Program:**\n"
            for program, avg in sorted(program_stats.items(), key=lambda x: x[1], reverse=True):
                response += f"• {program.replace('_', ' ').title()}: {avg:.2f}\n"
        
        # Add insights
        if avg_gpa >= 3.5:
            response += "\n💡 **Insight:** Strong applicant pool with high academic performance!"
        elif avg_gpa >= 3.0:
            response += "\n💡 **Insight:** Solid applicant pool meeting basic requirements."
        else:
            response += "\n💡 **Insight:** Many applicants below typical 3.0 GPA threshold."
        
        return response
    
    def _handle_decision_stats(self, user_input: str) -> str:
        """Handle decision-related queries"""
        
        decisions = [app.get('decision', 'Unknown') for app in self.applications_cache]
        decision_counts = Counter(decisions)
        total = len(decisions)
        
        if total == 0:
            return "No decision data available."
        
        response = f"📈 **Decision Statistics** ({total} applications):\n\n"
        
        # Sort by importance
        decision_order = ['ACCEPT_ACADEMIC', 'REVIEW', 'REJECT_ACADEMIC', 'ABSTAIN']
        
        for decision in decision_order:
            count = decision_counts.get(decision, 0)
            percentage = (count / total) * 100
            
            emoji = {
                'ACCEPT_ACADEMIC': '✅',
                'REVIEW': '⏳',
                'REJECT_ACADEMIC': '❌', 
                'ABSTAIN': '⏸️'
            }.get(decision, '❓')
            
            response += f"• {emoji} **{decision.replace('_', ' ').title()}:** {count} ({percentage:.1f}%)\n"
        
        # Calculate key metrics
        accept_rate = (decision_counts.get('ACCEPT_ACADEMIC', 0) / total) * 100
        abstention_rate = (decision_counts.get('ABSTAIN', 0) / total) * 100
        human_review_rate = ((decision_counts.get('REVIEW', 0) + decision_counts.get('ABSTAIN', 0)) / total) * 100
        
        response += f"""
**Key Metrics:**
• **Accept Rate:** {accept_rate:.1f}%
• **Human Review Rate:** {human_review_rate:.1f}%
• **Abstention Rate:** {abstention_rate:.1f}%
"""
        
        # Add insights
        if abstention_rate > 20:
            response += "\n⚠️ **Alert:** High abstention rate - consider reviewing confidence thresholds."
        elif accept_rate > 60:
            response += "\n💡 **Insight:** High acceptance rate indicates strong applicant pool or low thresholds."
        
        return response
    
    def _handle_confidence_stats(self, user_input: str) -> str:
        """Handle confidence-related queries"""
        
        confidences = [app.get('confidence', 0) for app in self.applications_cache if app.get('confidence', 0) > 0]
        
        if not confidences:
            return "No confidence data available."
        
        avg_confidence = statistics.mean(confidences)
        median_confidence = statistics.median(confidences)
        
        # Categorize confidence levels
        high_conf = len([c for c in confidences if c >= 0.8])
        medium_conf = len([c for c in confidences if 0.6 <= c < 0.8])
        low_conf = len([c for c in confidences if c < 0.6])
        
        response = f"""🎯 **Confidence Analysis** ({len(confidences)} applications):

• **Average Confidence:** {avg_confidence:.2f}
• **Median Confidence:** {median_confidence:.2f}

**Distribution:**
• **High Confidence (≥0.8):** {high_conf} ({high_conf/len(confidences)*100:.1f}%)
• **Medium Confidence (0.6-0.8):** {medium_conf} ({medium_conf/len(confidences)*100:.1f}%)
• **Low Confidence (<0.6):** {low_conf} ({low_conf/len(confidences)*100:.1f}%)
"""
        
        # Confidence by decision type
        confidence_by_decision = {}
        for app in self.applications_cache:
            decision = app.get('decision', 'Unknown')
            confidence = app.get('confidence', 0)
            if confidence > 0:
                if decision not in confidence_by_decision:
                    confidence_by_decision[decision] = []
                confidence_by_decision[decision].append(confidence)
        
        if confidence_by_decision:
            response += "\n**Average Confidence by Decision:**\n"
            for decision, confs in confidence_by_decision.items():
                avg_conf = statistics.mean(confs)
                response += f"• {decision}: {avg_conf:.2f}\n"
        
        return response
    
    def _handle_warning_stats(self, user_input: str) -> str:
        """Handle warning-related queries"""
        
        all_warnings = []
        apps_with_warnings = 0
        
        for app in self.applications_cache:
            warnings = app.get('warnings', [])
            if warnings:
                apps_with_warnings += 1
                all_warnings.extend(warnings)
        
        if not all_warnings:
            return "✅ Great news! No warnings found in processed applications."
        
        warning_counts = Counter(all_warnings)
        total_apps = len(self.applications_cache)
        
        response = f"""⚠️ **Warning Analysis**:

• **Applications with warnings:** {apps_with_warnings}/{total_apps} ({apps_with_warnings/total_apps*100:.1f}%)
• **Total warnings:** {len(all_warnings)}

**Most Common Warnings:**
"""
        
        for warning, count in warning_counts.most_common(5):
            response += f"• {warning} ({count} times)\n"
        
        # Add recommendations
        if apps_with_warnings / total_apps > 0.3:
            response += "\n💡 **Recommendation:** High warning rate - consider reviewing OCR quality or document formats."
        
        return response
    
    def _handle_program_stats(self, user_input: str) -> str:
        """Handle program-related queries"""
        
        programs = [app.get('program', 'general') for app in self.applications_cache]
        program_counts = Counter(programs)
        
        if not programs:
            return "No program data available."
        
        response = f"🏫 **Program Distribution** ({len(programs)} applications):\n\n"
        
        # Program statistics
        for program, count in program_counts.most_common():
            percentage = (count / len(programs)) * 100
            response += f"• **{program.replace('_', ' ').title()}:** {count} ({percentage:.1f}%)\n"
        
        # Accept rates by program
        program_decisions = {}
        for app in self.applications_cache:
            program = app.get('program', 'general')
            decision = app.get('decision', 'Unknown')
            
            if program not in program_decisions:
                program_decisions[program] = {'total': 0, 'accepted': 0}
            
            program_decisions[program]['total'] += 1
            if decision == 'ACCEPT_ACADEMIC':
                program_decisions[program]['accepted'] += 1
        
        response += "\n**Accept Rates by Program:**\n"
        for program, stats in program_decisions.items():
            if stats['total'] > 0:
                accept_rate = (stats['accepted'] / stats['total']) * 100
                response += f"• {program.replace('_', ' ').title()}: {accept_rate:.1f}% ({stats['accepted']}/{stats['total']})\n"
        
        return response
    
    def _handle_processing_time(self, user_input: str) -> str:
        """Handle processing time queries"""
        
        processing_times = []
        for app in self.applications_cache:
            time = app.get('processing_info', {}).get('processing_time_seconds', 0)
            if time > 0:
                processing_times.append(time)
        
        if not processing_times:
            return "No processing time data available."
        
        avg_time = statistics.mean(processing_times)
        median_time = statistics.median(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        # Calculate throughput
        throughput_per_hour = 3600 / avg_time if avg_time > 0 else 0
        
        # Time savings estimate
        manual_time = 20 * 60  # 20 minutes manual review
        time_saved = manual_time - avg_time
        savings_percentage = (time_saved / manual_time) * 100
        
        response = f"""⚡ **Processing Performance** ({len(processing_times)} applications):

• **Average Time:** {avg_time:.1f} seconds
• **Median Time:** {median_time:.1f} seconds
• **Range:** {min_time:.1f}s - {max_time:.1f}s

**Efficiency Metrics:**
• **Throughput:** {throughput_per_hour:.0f} applications/hour
• **Time Savings:** {savings_percentage:.1f}% vs manual review
• **Time Saved per 100 apps:** {time_saved * 100 / 3600:.1f} hours

"""
        
        if avg_time < 30:
            response += "🚀 **Excellent:** Very fast processing times!"
        elif avg_time < 60:
            response += "✅ **Good:** Reasonable processing performance."
        else:
            response += "⚠️ **Note:** Processing times could be optimized."
        
        return response
    
    def _handle_unknown_query(self, user_input: str) -> str:
        """Handle queries that don't match known patterns"""
        
        # Try to be helpful by suggesting related information
        suggestions = []
        
        if any(word in user_input for word in ["number", "count", "how many"]):
            suggestions.append("📊 Try asking about decision counts or program distribution")
        
        if any(word in user_input for word in ["best", "worst", "highest", "lowest"]):
            suggestions.append("📈 Try asking about average GPA or accept rates by program")
        
        if any(word in user_input for word in ["time", "speed", "fast", "slow"]):
            suggestions.append("⚡ Try asking about processing times or throughput")
        
        base_response = "I'm not sure how to answer that specific question."
        
        if suggestions:
            return f"{base_response}\n\n💡 **Suggestions:**\n" + "\n".join(f"• {s}" for s in suggestions)
        
        return f"""{base_response}

💡 **Try asking about:**
• Application statistics (GPA, decisions, confidence)
• Program comparisons
• Processing performance
• Warning analysis

Or type "help" for more examples!"""


if __name__ == "__main__":
    # Test the chat bot
    bot = ChatBot()
    
    test_queries = [
        "Hello",
        "What's the average GPA?",
        "How many applications were accepted?",
        "Show me confidence stats",
        "What are common warnings?",
        "Help"
    ]
    
    for query in test_queries:
        print(f"User: {query}")
        response = bot.get_response(query)
        print(f"Bot: {response}\n")
        print("-" * 50)
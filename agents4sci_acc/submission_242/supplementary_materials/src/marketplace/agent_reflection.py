"""
Simplified Agent Reflection System

This module provides basic reflection functionality for marketplace agents.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ReflectionSession:
    """Simple reflection session record"""
    agent_id: str
    agent_type: str  # "freelancer" or "client"
    timestamp: datetime
    performance_summary: str
    key_insights: List[str]
    planned_adjustments: List[str]
    
    @property
    def learnings(self) -> List[str]:
        """Alias for key_insights for backward compatibility."""
        return self.key_insights

class AgentReflectionManager:
    """Manages agent reflection sessions"""
    
    def __init__(self):
        self._reflection_history: Dict[str, List[ReflectionSession]] = {}
    
    def create_reflection_session(
        self,
        agent_id: str,
        agent_type: str,
        recent_activities: List[Dict]
    ) -> ReflectionSession:
        """Create a new reflection session for an agent"""
        
        # Calculate basic performance metrics
        performance_summary = self._get_performance_summary(agent_type, recent_activities)
        
        # Generate simple insights based on activities
        key_insights = self._generate_insights(agent_type, recent_activities)
        
        # Create basic adjustment plan
        planned_adjustments = self._plan_adjustments(agent_type, recent_activities)
        
        # Create session
        session = ReflectionSession(
            agent_id=agent_id,
            agent_type=agent_type,
            timestamp=datetime.now(),
            performance_summary=performance_summary,
            key_insights=key_insights,
            planned_adjustments=planned_adjustments
        )
        
        # Store in history
        if agent_id not in self._reflection_history:
            self._reflection_history[agent_id] = []
        self._reflection_history[agent_id].append(session)
        
        return session
    
    def get_agent_reflections(
        self,
        agent_id: str,
        days: Optional[int] = None
    ) -> List[ReflectionSession]:
        """Get reflection history for an agent"""
        sessions = self._reflection_history.get(agent_id, [])
        if days is not None:
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(days=days)
            sessions = [s for s in sessions if s.timestamp >= cutoff]
        return sessions
    
    def get_learning_progress(self, agent_id: str) -> Dict:
        """Get simple learning progress for an agent"""
        sessions = self._reflection_history.get(agent_id, [])
        if not sessions:
            return {'total_reflections': 0, 'total_learnings': 0, 'recent_insights': [], 'learning_trend': 'stable', 'top_insights': []}
        
        # Calculate total learnings across all sessions
        total_learnings = sum(len(session.key_insights) for session in sessions)
        
        # Simple learning trend based on recent vs older sessions
        if len(sessions) >= 2:
            recent_insights = len(sessions[-1].key_insights)
            previous_insights = len(sessions[-2].key_insights)
            learning_trend = 'improving' if recent_insights > previous_insights else 'stable'
        else:
            learning_trend = 'stable'
        
        # Get top insights from all sessions
        all_insights = []
        for session in sessions:
            all_insights.extend(session.key_insights)
        top_insights = all_insights[:5]  # Top 5 insights
        
        return {
            'total_reflections': len(sessions),
            'total_learnings': total_learnings,
            'recent_insights': sessions[-1].key_insights if sessions else [],
            'latest_adjustments': sessions[-1].planned_adjustments if sessions else [],
            'learning_trend': learning_trend,
            'top_insights': top_insights
        }
    
    def _get_performance_summary(self, agent_type: str, activities: List[Dict]) -> str:
        """Generate a simple performance summary"""
        if agent_type == "freelancer":
            total_bids = len([a for a in activities if a.get('type') == 'bid'])
            successful_bids = len([a for a in activities if a.get('type') == 'bid' and a.get('outcome') == 'success'])
            success_rate = successful_bids / total_bids if total_bids > 0 else 0
            return f"Made {total_bids} bids with {success_rate:.1%} success rate"
        else:  # client
            total_posts = len([a for a in activities if a.get('type') == 'job_post'])
            filled_posts = len([a for a in activities if a.get('type') == 'job_post' and a.get('outcome') == 'filled'])
            fill_rate = filled_posts / total_posts if total_posts > 0 else 0
            return f"Posted {total_posts} jobs with {fill_rate:.1%} fill rate"
    
    def _generate_insights(self, agent_type: str, activities: List[Dict]) -> List[str]:
        """Generate simple insights from activities"""
        insights = []
        
        if agent_type == "freelancer":
            failed_bids = [a for a in activities if a.get('type') == 'bid' and a.get('outcome') != 'success']
            if len(failed_bids) > 2:
                insights.append("High number of unsuccessful bids - may need to adjust strategy")
            if any('rate_too_high' in a.get('failure_reason', '') for a in failed_bids):
                insights.append("Pricing may be too aggressive for current market")
        else:  # client
            unfilled_posts = [a for a in activities if a.get('type') == 'job_post' and a.get('outcome') != 'filled']
            if len(unfilled_posts) > 1:
                insights.append("Multiple unfilled positions - may need to adjust requirements or budget")
        
        return insights[:3]  # Keep it simple - max 3 insights
    
    def _plan_adjustments(self, agent_type: str, activities: List[Dict]) -> List[str]:
        """Plan simple adjustments based on performance"""
        adjustments = []
        
        if agent_type == "freelancer":
            failed_bids = [a for a in activities if a.get('type') == 'bid' and a.get('outcome') != 'success']
            if len(failed_bids) > 2:
                adjustments.append("Consider more competitive pricing")
                adjustments.append("Focus on jobs that better match skills")
        else:  # client
            unfilled_posts = [a for a in activities if a.get('type') == 'job_post' and a.get('outcome') != 'filled']
            if len(unfilled_posts) > 1:
                adjustments.append("Review job requirements for flexibility")
                adjustments.append("Consider increasing budget for key positions")
        
        return adjustments[:2]  # Keep it simple - max 2 adjustments

# Global instance for easy access
reflection_manager = AgentReflectionManager()
#!/usr/bin/env python3
"""
Enhanced Survey Evaluator with Multi-Dimensional Scoring
Based on lessons learned from existing evaluations
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

class EnhancedSurveyEvaluator:
    """
    Comprehensive survey evaluation with 12 dimensions instead of 5
    """
    
    def __init__(self):
        self.dimensions = {
            # Core Quality Dimensions (60% total weight)
            'citation_coverage': {'weight': 0.15, 'category': 'core'},
            'accuracy': {'weight': 0.15, 'category': 'core'},
            'synthesis_quality': {'weight': 0.15, 'category': 'core'},
            'organization': {'weight': 0.15, 'category': 'core'},
            
            # Writing Quality Dimensions (20% total weight)
            'readability': {'weight': 0.05, 'category': 'writing'},
            'academic_rigor': {'weight': 0.05, 'category': 'writing'},
            'clarity': {'weight': 0.05, 'category': 'writing'},
            'coherence': {'weight': 0.05, 'category': 'writing'},
            
            # Content Depth Dimensions (20% total weight)
            'comprehensiveness': {'weight': 0.05, 'category': 'content'},
            'critical_analysis': {'weight': 0.05, 'category': 'content'},
            'novelty_insights': {'weight': 0.05, 'category': 'content'},
            'future_directions': {'weight': 0.05, 'category': 'content'}
        }
        
    def evaluate_survey(self, survey_path: str, papers_path: str, clusters_path: str) -> Dict:
        """Perform comprehensive evaluation"""
        
        # Load files
        with open(survey_path, 'r') as f:
            survey_text = f.read()
        
        with open(papers_path, 'r') as f:
            papers = json.load(f)
            if 'papers' in papers:
                papers = papers['papers']
        
        with open(clusters_path, 'r') as f:
            clusters = json.load(f)
        
        # Calculate all dimension scores
        scores = {}
        
        # 1. Citation Coverage (15%)
        scores['citation_coverage'] = self._evaluate_citation_coverage(survey_text, papers)
        
        # 2. Accuracy (15%)
        scores['accuracy'] = self._evaluate_accuracy(survey_text, papers)
        
        # 3. Synthesis Quality (15%)
        scores['synthesis_quality'] = self._evaluate_synthesis(survey_text)
        
        # 4. Organization (15%)
        scores['organization'] = self._evaluate_organization(survey_text, clusters)
        
        # 5. Readability (5%)
        scores['readability'] = self._evaluate_readability(survey_text)
        
        # 6. Academic Rigor (5%)
        scores['academic_rigor'] = self._evaluate_academic_rigor(survey_text)
        
        # 7. Clarity (5%)
        scores['clarity'] = self._evaluate_clarity(survey_text)
        
        # 8. Coherence (5%)
        scores['coherence'] = self._evaluate_coherence(survey_text)
        
        # 9. Comprehensiveness (5%)
        scores['comprehensiveness'] = self._evaluate_comprehensiveness(survey_text, clusters)
        
        # 10. Critical Analysis (5%)
        scores['critical_analysis'] = self._evaluate_critical_analysis(survey_text)
        
        # 11. Novelty & Insights (5%)
        scores['novelty_insights'] = self._evaluate_novelty(survey_text)
        
        # 12. Future Directions (5%)
        scores['future_directions'] = self._evaluate_future_directions(survey_text)
        
        # Calculate weighted overall score
        overall_score = sum(
            scores[dim]['score'] * self.dimensions[dim]['weight'] 
            for dim in self.dimensions
        )
        
        # Calculate category scores
        category_scores = {}
        for category in ['core', 'writing', 'content']:
            dims = [d for d, info in self.dimensions.items() if info['category'] == category]
            category_scores[category] = np.mean([scores[d]['score'] for d in dims])
        
        # Generate comprehensive evaluation
        evaluation = {
            'metadata': {
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluator_version': '2.0_enhanced',
                'survey_path': survey_path,
                'word_count': len(survey_text.split()),
                'total_papers': len(papers),
                'clusters_count': len(clusters.get('clusters', []))
            },
            'overall_score': round(overall_score, 2),
            'category_scores': {
                'core_quality': round(category_scores['core'], 2),
                'writing_quality': round(category_scores['writing'], 2),
                'content_depth': round(category_scores['content'], 2)
            },
            'dimension_scores': scores,
            'grade': self._calculate_grade(overall_score),
            'publication_readiness': self._assess_publication_readiness(overall_score, scores),
            'strengths': self._identify_strengths(scores),
            'weaknesses': self._identify_weaknesses(scores),
            'recommendations': self._generate_recommendations(scores)
        }
        
        return evaluation
    
    def _evaluate_citation_coverage(self, survey: str, papers: List) -> Dict:
        """Evaluate citation coverage with detailed metrics"""
        
        # Extract citations from survey
        citation_pattern = r'\[([^\]]+?),?\s*(\d{4})[a-z]?\]'
        citations = re.findall(citation_pattern, survey)
        
        # Match citations to papers
        cited_papers = set()
        for author, year in citations:
            author_clean = author.split(',')[0].split(' ')[0].lower()
            year_int = int(year)
            
            for i, paper in enumerate(papers):
                if 'authors' in paper and paper['authors']:
                    first_author = paper['authors'][0].split(' ')[-1].lower()
                    paper_year = paper.get('year', 0)
                    
                    if first_author == author_clean and paper_year == year_int:
                        cited_papers.add(i)
        
        coverage_rate = len(cited_papers) / len(papers) if papers else 0
        
        # Calculate score based on coverage
        if coverage_rate >= 0.8:
            score = 10.0
        elif coverage_rate >= 0.6:
            score = 8.5
        elif coverage_rate >= 0.5:
            score = 7.5
        elif coverage_rate >= 0.4:
            score = 6.5
        elif coverage_rate >= 0.3:
            score = 5.5
        else:
            score = 3.0 + (coverage_rate * 10)
        
        return {
            'score': score,
            'details': {
                'papers_cited': len(cited_papers),
                'total_papers': len(papers),
                'coverage_percentage': round(coverage_rate * 100, 1),
                'unique_citations': len(citations),
                'citation_density': round(len(citations) / (len(survey.split()) / 100), 1)
            }
        }
    
    def _evaluate_accuracy(self, survey: str, papers: List) -> Dict:
        """Evaluate factual accuracy and citation correctness"""
        
        # Check for citation format consistency
        good_format = len(re.findall(r'\[[A-Za-z]+(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\]', survey))
        bad_format = len(re.findall(r'\([A-Za-z]+.*\d{4}\)', survey))  # Wrong format
        
        format_score = 10.0 if bad_format == 0 else max(5.0, 10.0 - bad_format)
        
        # Check for statistical claims with citations
        stats_pattern = r'\d+(?:\.\d+)?%|\d+(?:,\d{3})+\s+papers?|\d+\s+studies'
        stats_claims = re.findall(stats_pattern, survey)
        
        # Check if claims near citations
        stats_with_citations = 0
        for claim in stats_claims[:10]:  # Sample first 10
            claim_pos = survey.find(claim)
            nearby_citation = survey[max(0, claim_pos-50):claim_pos+100].count('[')
            if nearby_citation > 0:
                stats_with_citations += 1
        
        claim_score = 10.0 if not stats_claims else (stats_with_citations / min(10, len(stats_claims))) * 10
        
        final_score = (format_score * 0.6 + claim_score * 0.4)
        
        return {
            'score': final_score,
            'details': {
                'citation_format_score': format_score,
                'claim_accuracy_score': claim_score,
                'total_citations': good_format,
                'format_errors': bad_format,
                'statistical_claims': len(stats_claims),
                'supported_claims': stats_with_citations
            }
        }
    
    def _evaluate_synthesis(self, survey: str) -> Dict:
        """Evaluate synthesis quality"""
        
        # Look for synthesis indicators
        synthesis_phrases = [
            'in contrast', 'compared to', 'while', 'whereas', 'however',
            'similarly', 'likewise', 'on the other hand', 'conversely',
            'builds upon', 'extends', 'differs from', 'agrees with',
            'common thread', 'pattern emerges', 'trend shows', 'consensus'
        ]
        
        synthesis_count = sum(survey.lower().count(phrase) for phrase in synthesis_phrases)
        
        # Check for comparative structures
        comparative_sections = len(re.findall(r'compar\w+|contrast\w+|similar\w+|differ\w+', survey, re.I))
        
        # Check for "Author did X" pattern (bad)
        list_pattern = len(re.findall(r'[A-Z][a-z]+\s+et\s+al\.\s+\[\d{4}\]\s+(propose[sd]?|present[sd]?|introduce[sd]?|develop[sd]?)', survey))
        
        # Calculate score
        synthesis_density = synthesis_count / (len(survey.split()) / 100)
        
        if synthesis_density >= 5 and list_pattern < 5:
            score = 9.0
        elif synthesis_density >= 3:
            score = 7.5
        elif synthesis_density >= 2:
            score = 6.5
        else:
            score = 5.0
        
        score = max(3.0, score - (list_pattern * 0.2))
        
        return {
            'score': min(10.0, score),
            'details': {
                'synthesis_phrases_found': synthesis_count,
                'synthesis_density': round(synthesis_density, 2),
                'comparative_sections': comparative_sections,
                'list_like_patterns': list_pattern
            }
        }
    
    def _evaluate_organization(self, survey: str, clusters: Dict) -> Dict:
        """Evaluate structural organization"""
        
        # Check for required sections
        required_sections = ['abstract', 'introduction', 'conclusion', 'references']
        found_sections = sum(1 for section in required_sections if section in survey.lower())
        
        # Check cluster coverage
        num_clusters = len(clusters.get('clusters', []))
        cluster_sections = 0
        for cluster in clusters.get('clusters', []):
            if cluster.get('name', '').lower() in survey.lower():
                cluster_sections += 1
        
        cluster_coverage = cluster_sections / num_clusters if num_clusters > 0 else 0
        
        # Check section balance (using headers)
        headers = re.findall(r'^#{1,3}\s+(.+)$', survey, re.MULTILINE)
        
        # Calculate score
        structure_score = (found_sections / len(required_sections)) * 10
        cluster_score = cluster_coverage * 10
        
        final_score = structure_score * 0.5 + cluster_score * 0.5
        
        return {
            'score': final_score,
            'details': {
                'required_sections_found': found_sections,
                'total_required': len(required_sections),
                'clusters_covered': cluster_sections,
                'total_clusters': num_clusters,
                'total_sections': len(headers)
            }
        }
    
    def _evaluate_readability(self, survey: str) -> Dict:
        """Evaluate readability metrics"""
        
        sentences = survey.split('.')
        words = survey.split()
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Check for overly long sentences
        long_sentences = sum(1 for s in sentences if len(s.split()) > 40)
        
        # Technical term usage (should be defined)
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', survey))  # Acronyms
        
        # Calculate score
        if 15 <= avg_sentence_length <= 25 and long_sentences < 10:
            score = 9.0
        elif 10 <= avg_sentence_length <= 30:
            score = 7.5
        else:
            score = 6.0
        
        return {
            'score': score,
            'details': {
                'avg_sentence_length': round(avg_sentence_length, 1),
                'long_sentences': long_sentences,
                'technical_terms': technical_terms,
                'total_sentences': len(sentences)
            }
        }
    
    def _evaluate_academic_rigor(self, survey: str) -> Dict:
        """Evaluate academic writing standards"""
        
        # Check for informal language
        informal_words = ['really', 'very', 'stuff', 'things', 'basically', 'actually', 'just']
        informal_count = sum(survey.lower().count(word) for word in informal_words)
        
        # Check for proper academic phrases
        academic_phrases = [
            'furthermore', 'moreover', 'consequently', 'therefore',
            'nevertheless', 'nonetheless', 'specifically', 'particularly'
        ]
        academic_count = sum(survey.lower().count(phrase) for phrase in academic_phrases)
        
        # Check for hedging language (good in academic writing)
        hedging = ['may', 'might', 'could', 'possibly', 'potentially', 'suggest', 'indicate']
        hedging_count = sum(survey.lower().count(word) for word in hedging)
        
        # Calculate score
        score = 7.0
        score += min(2.0, academic_count * 0.1)
        score -= min(2.0, informal_count * 0.2)
        score += min(1.0, hedging_count * 0.05)
        
        return {
            'score': min(10.0, max(3.0, score)),
            'details': {
                'informal_language_instances': informal_count,
                'academic_phrases_used': academic_count,
                'hedging_language_used': hedging_count
            }
        }
    
    def _evaluate_clarity(self, survey: str) -> Dict:
        """Evaluate clarity of writing"""
        
        # Check for complex nested sentences
        nested_parens = len(re.findall(r'\([^)]*\([^)]*\)', survey))
        
        # Check for clear topic sentences (paragraphs starting with clear statements)
        paragraphs = [p for p in survey.split('\n\n') if len(p) > 100]
        clear_starts = sum(1 for p in paragraphs if p[0].isupper() and '.' in p[:150])
        
        clarity_ratio = clear_starts / len(paragraphs) if paragraphs else 0
        
        # Calculate score
        score = 8.0
        score -= min(2.0, nested_parens * 0.5)
        score += clarity_ratio * 2.0
        
        return {
            'score': min(10.0, max(3.0, score)),
            'details': {
                'nested_complexity': nested_parens,
                'clear_paragraphs': clear_starts,
                'total_paragraphs': len(paragraphs),
                'clarity_ratio': round(clarity_ratio, 2)
            }
        }
    
    def _evaluate_coherence(self, survey: str) -> Dict:
        """Evaluate logical flow and coherence"""
        
        # Check for transition words between paragraphs
        transitions = [
            'however', 'furthermore', 'moreover', 'additionally',
            'in addition', 'consequently', 'therefore', 'thus',
            'nevertheless', 'meanwhile', 'subsequently', 'finally'
        ]
        
        transition_count = sum(survey.lower().count(t) for t in transitions)
        
        # Check for section connections
        sections = re.split(r'^#{1,3}\s+', survey, flags=re.MULTILINE)
        connected_sections = sum(1 for s in sections if any(t in s.lower()[:200] for t in transitions))
        
        connection_ratio = connected_sections / len(sections) if sections else 0
        
        # Calculate score
        transition_density = transition_count / (len(survey.split()) / 100)
        
        score = 6.0 + min(4.0, transition_density * 0.8 + connection_ratio * 3.2)
        
        return {
            'score': score,
            'details': {
                'transition_words': transition_count,
                'transition_density': round(transition_density, 2),
                'connected_sections': connected_sections,
                'total_sections': len(sections)
            }
        }
    
    def _evaluate_comprehensiveness(self, survey: str, clusters: Dict) -> Dict:
        """Evaluate comprehensive coverage of the topic"""
        
        # Check if all clusters are discussed
        clusters_mentioned = 0
        cluster_names = [c.get('name', '') for c in clusters.get('clusters', [])]
        
        for name in cluster_names:
            if name.lower() in survey.lower():
                clusters_mentioned += 1
        
        cluster_coverage = clusters_mentioned / len(cluster_names) if cluster_names else 0
        
        # Check for discussion of limitations
        has_limitations = 'limitation' in survey.lower() or 'challenge' in survey.lower()
        
        # Check for discussion of methodology
        has_methodology = 'method' in survey.lower() or 'approach' in survey.lower()
        
        # Calculate score
        score = cluster_coverage * 7.0
        if has_limitations:
            score += 1.5
        if has_methodology:
            score += 1.5
        
        return {
            'score': min(10.0, score),
            'details': {
                'clusters_discussed': clusters_mentioned,
                'total_clusters': len(cluster_names),
                'discusses_limitations': has_limitations,
                'discusses_methodology': has_methodology
            }
        }
    
    def _evaluate_critical_analysis(self, survey: str) -> Dict:
        """Evaluate depth of critical analysis"""
        
        # Look for critical analysis indicators
        critical_phrases = [
            'limitation', 'drawback', 'challenge', 'issue', 'problem',
            'critique', 'weakness', 'strength', 'advantage', 'disadvantage',
            'trade-off', 'however', 'despite', 'although', 'while'
        ]
        
        critical_count = sum(survey.lower().count(phrase) for phrase in critical_phrases)
        
        # Check for balanced analysis (both pros and cons)
        has_pros = any(word in survey.lower() for word in ['advantage', 'strength', 'benefit'])
        has_cons = any(word in survey.lower() for word in ['disadvantage', 'limitation', 'drawback'])
        is_balanced = has_pros and has_cons
        
        # Calculate score
        critical_density = critical_count / (len(survey.split()) / 100)
        
        score = min(8.0, critical_density * 1.5)
        if is_balanced:
            score += 2.0
        
        return {
            'score': min(10.0, score),
            'details': {
                'critical_phrases_found': critical_count,
                'critical_density': round(critical_density, 2),
                'balanced_analysis': is_balanced,
                'discusses_pros': has_pros,
                'discusses_cons': has_cons
            }
        }
    
    def _evaluate_novelty(self, survey: str) -> Dict:
        """Evaluate novel insights and contributions"""
        
        # Look for insight indicators
        insight_phrases = [
            'we find', 'we observe', 'our analysis shows', 'this suggests',
            'interestingly', 'surprisingly', 'notably', 'importantly',
            'key insight', 'main finding', 'reveals', 'demonstrates'
        ]
        
        insight_count = sum(survey.lower().count(phrase) for phrase in insight_phrases)
        
        # Check for future research directions
        has_future = 'future' in survey.lower() and 'research' in survey.lower()
        
        # Check for identified gaps
        has_gaps = 'gap' in survey.lower() or 'missing' in survey.lower()
        
        # Calculate score
        insight_density = insight_count / (len(survey.split()) / 100)
        
        score = min(7.0, insight_density * 2.0)
        if has_future:
            score += 1.5
        if has_gaps:
            score += 1.5
        
        return {
            'score': min(10.0, score),
            'details': {
                'insight_phrases_found': insight_count,
                'insight_density': round(insight_density, 2),
                'discusses_future_research': has_future,
                'identifies_gaps': has_gaps
            }
        }
    
    def _evaluate_future_directions(self, survey: str) -> Dict:
        """Evaluate quality of future directions section"""
        
        # Check if future directions section exists
        has_section = 'future' in survey.lower() and ('direction' in survey.lower() or 'work' in survey.lower())
        
        # Count specific future research suggestions
        future_patterns = [
            'future work', 'future research', 'could be', 'should investigate',
            'remains to be', 'open question', 'unexplored', 'promising direction'
        ]
        
        future_count = sum(survey.lower().count(pattern) for pattern in future_patterns)
        
        # Check for specific vs generic suggestions
        specific_suggestions = len(re.findall(r'future[^.]*specific[^.]*\.', survey, re.I))
        
        # Calculate score
        if not has_section:
            score = 3.0
        else:
            score = 6.0 + min(4.0, future_count * 0.5)
        
        return {
            'score': min(10.0, score),
            'details': {
                'has_future_section': has_section,
                'future_research_mentions': future_count,
                'specific_suggestions': specific_suggestions
            }
        }
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade"""
        if score >= 9.0:
            return 'A'
        elif score >= 8.5:
            return 'A-'
        elif score >= 8.0:
            return 'B+'
        elif score >= 7.5:
            return 'B'
        elif score >= 7.0:
            return 'B-'
        elif score >= 6.5:
            return 'C+'
        elif score >= 6.0:
            return 'C'
        elif score >= 5.5:
            return 'C-'
        else:
            return 'D'
    
    def _assess_publication_readiness(self, score: float, dimension_scores: Dict) -> Dict:
        """Assess publication readiness"""
        
        if score >= 8.5:
            readiness = 'Ready for top-tier venues'
            venues = ['ACM Computing Surveys', 'Nature Machine Intelligence', 'NeurIPS']
        elif score >= 7.5:
            readiness = 'Ready for good conferences/journals'
            venues = ['ACL', 'EMNLP', 'ICML workshops']
        elif score >= 6.5:
            readiness = 'Ready for workshops with minor revisions'
            venues = ['Workshop papers', 'ArXiv preprint']
        else:
            readiness = 'Requires significant revision'
            venues = ['Internal review', 'Technical report']
        
        return {
            'readiness_level': readiness,
            'suggested_venues': venues,
            'critical_improvements_needed': score < 7.0,
            'estimated_revision_time': '1-2 days' if score >= 7.5 else '3-5 days' if score >= 6.5 else '1-2 weeks'
        }
    
    def _identify_strengths(self, scores: Dict) -> List[str]:
        """Identify top strengths"""
        strengths = []
        
        for dim, data in scores.items():
            if data['score'] >= 8.0:
                strengths.append(f"Excellent {dim.replace('_', ' ')}: {data['score']:.1f}/10")
            elif data['score'] >= 7.0:
                strengths.append(f"Good {dim.replace('_', ' ')}: {data['score']:.1f}/10")
        
        return strengths[:5]  # Top 5 strengths
    
    def _identify_weaknesses(self, scores: Dict) -> List[str]:
        """Identify main weaknesses"""
        weaknesses = []
        
        for dim, data in scores.items():
            if data['score'] < 5.0:
                weaknesses.append(f"Poor {dim.replace('_', ' ')}: {data['score']:.1f}/10")
            elif data['score'] < 6.0:
                weaknesses.append(f"Weak {dim.replace('_', ' ')}: {data['score']:.1f}/10")
        
        return weaknesses[:5]  # Top 5 weaknesses
    
    def _generate_recommendations(self, scores: Dict) -> List[Dict]:
        """Generate specific recommendations"""
        recommendations = []
        
        # Check each dimension and provide targeted recommendations
        if scores['citation_coverage']['score'] < 7.0:
            recommendations.append({
                'priority': 'HIGH',
                'dimension': 'citation_coverage',
                'recommendation': f"Increase citation coverage from {scores['citation_coverage']['details']['coverage_percentage']}% to at least 50%",
                'expected_impact': '+1.5 to overall score'
            })
        
        if scores['synthesis_quality']['score'] < 7.0:
            recommendations.append({
                'priority': 'HIGH',
                'dimension': 'synthesis_quality',
                'recommendation': 'Add more comparative analysis and reduce sequential paper listing',
                'expected_impact': '+1.0 to overall score'
            })
        
        if scores['critical_analysis']['score'] < 6.0:
            recommendations.append({
                'priority': 'MEDIUM',
                'dimension': 'critical_analysis',
                'recommendation': 'Include discussion of limitations and trade-offs for each approach',
                'expected_impact': '+0.5 to overall score'
            })
        
        if scores['future_directions']['score'] < 6.0:
            recommendations.append({
                'priority': 'MEDIUM',
                'dimension': 'future_directions',
                'recommendation': 'Add dedicated future directions section with specific research questions',
                'expected_impact': '+0.3 to overall score'
            })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations[:5]  # Top 5 recommendations


def evaluate_all_surveys():
    """Evaluate all completed surveys with enhanced metrics"""
    
    evaluator = EnhancedSurveyEvaluator()
    base_path = Path('/path/to/project/output')
    
    results = []
    
    # Find all directories with completed surveys
    for survey_dir in base_path.iterdir():
        if not survey_dir.is_dir():
            continue
        
        output_dir = survey_dir / 'output'
        if not output_dir.exists():
            continue
        
        survey_file = output_dir / 'survey.md'
        papers_file = output_dir / 'papers.json'
        clusters_file = output_dir / 'clusters.json'
        
        # Check if all required files exist
        if all(f.exists() for f in [survey_file, papers_file, clusters_file]):
            print(f"\nEvaluating: {survey_dir.name}")
            
            try:
                # Perform enhanced evaluation
                evaluation = evaluator.evaluate_survey(
                    str(survey_file),
                    str(papers_file),
                    str(clusters_file)
                )
                
                # Save enhanced evaluation
                eval_file = output_dir / 'enhanced_evaluation.json'
                with open(eval_file, 'w') as f:
                    json.dump(evaluation, f, indent=2)
                
                print(f"  ✅ Overall Score: {evaluation['overall_score']}/10 (Grade: {evaluation['grade']})")
                print(f"  📊 Core Quality: {evaluation['category_scores']['core_quality']:.1f}")
                print(f"  ✍️ Writing Quality: {evaluation['category_scores']['writing_quality']:.1f}")
                print(f"  📚 Content Depth: {evaluation['category_scores']['content_depth']:.1f}")
                print(f"  💾 Saved to: {eval_file}")
                
                results.append({
                    'survey': survey_dir.name,
                    'score': evaluation['overall_score'],
                    'grade': evaluation['grade'],
                    'categories': evaluation['category_scores']
                })
                
            except Exception as e:
                print(f"  ❌ Error evaluating {survey_dir.name}: {str(e)}")
    
    # Save summary
    summary_file = base_path / 'enhanced_evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'evaluation_date': datetime.now().isoformat(),
            'total_surveys_evaluated': len(results),
            'average_score': round(np.mean([r['score'] for r in results]), 2) if results else 0,
            'surveys': results
        }, f, indent=2)
    
    print(f"\n📋 Summary saved to: {summary_file}")
    print(f"📈 Average Score: {np.mean([r['score'] for r in results]):.2f}/10" if results else "No surveys evaluated")
    
    return results


if __name__ == '__main__':
    print("🔍 Enhanced Survey Evaluator v2.0")
    print("=" * 60)
    results = evaluate_all_surveys()
    print("\n✅ Evaluation complete!")
    print(f"Total surveys evaluated: {len(results)}")
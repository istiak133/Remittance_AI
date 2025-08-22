"""
Advanced Pattern Recognition Engine for Economic Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class PatternMatcher:
    """Advanced pattern recognition and economic cycle analysis"""
    
    def __init__(self, data_engine):
        """Initialize with data engine"""
        self.analyzer = data_engine
        self.data = data_engine.data
        self.processed_data = data_engine.processed_data
        
        # Economic events database (hardcoded for MVP)
        self.economic_events = self._load_economic_timeline()
        self.pattern_library = self._build_pattern_library()
        
        logger.info("Pattern matcher initialized with economic timeline")
    
    def _load_economic_timeline(self) -> Dict[int, List[Dict]]:
        """Load major economic events affecting Bangladesh remittances"""
        return {
            1997: [{"event": "Asian Financial Crisis", "impact": "negative", "severity": "high"}],
            1998: [{"event": "Bangladesh Floods", "impact": "mixed", "severity": "medium"}],
            2001: [{"event": "9/11 Attacks", "impact": "negative", "severity": "medium"}],
            2007: [{"event": "Global Food Crisis Begin", "impact": "negative", "severity": "medium"}],
            2008: [{"event": "Global Financial Crisis", "impact": "negative", "severity": "extreme"}],
            2009: [{"event": "Post-Crisis Recovery Begin", "impact": "positive", "severity": "medium"}],
            2011: [{"event": "Arab Spring", "impact": "mixed", "severity": "medium"}],
            2012: [{"event": "Euro Crisis", "impact": "negative", "severity": "medium"}],
            2014: [{"event": "Oil Price Collapse", "impact": "mixed", "severity": "medium"}],
            2016: [{"event": "Brexit Referendum", "impact": "negative", "severity": "low"}],
            2017: [{"event": "Trump Immigration Policies", "impact": "negative", "severity": "medium"}],
            2019: [{"event": "US-China Trade War", "impact": "negative", "severity": "medium"}],
            2020: [{"event": "COVID-19 Pandemic Begin", "impact": "negative", "severity": "extreme"}],
            2021: [{"event": "COVID Recovery", "impact": "positive", "severity": "high"}],
            2022: [{"event": "Russia-Ukraine War", "impact": "mixed", "severity": "high"}],
            2023: [{"event": "Global Inflation Peak", "impact": "mixed", "severity": "medium"}],
            2024: [{"event": "Bangladesh Political Transition", "impact": "mixed", "severity": "medium"}]
        }
    
    def _build_pattern_library(self) -> Dict:
        """Build library of known economic patterns"""
        return {
            'crisis_recovery': {
                'description': 'Sharp decline followed by gradual recovery',
                'characteristics': ['negative_shock', 'gradual_recovery', 'new_equilibrium'],
                'typical_duration': '3-5 years',
                'examples': ['2008 Financial Crisis', 'COVID-19 Pandemic']
            },
            'gradual_growth': {
                'description': 'Steady, sustained growth over time',
                'characteristics': ['consistent_positive_growth', 'low_volatility', 'trend_following'],
                'typical_duration': '5+ years',
                'examples': ['2010-2019 Growth Period']
            },
            'volatile_growth': {
                'description': 'Growth with high year-to-year fluctuations',
                'characteristics': ['high_volatility', 'irregular_patterns', 'external_sensitivity'],
                'typical_duration': 'Variable',
                'examples': ['1995-2005 Early Period']
            },
            'boom_cycle': {
                'description': 'Rapid growth followed by moderation',
                'characteristics': ['rapid_acceleration', 'peak_formation', 'moderation'],
                'typical_duration': '2-4 years',
                'examples': ['2005-2008 Boom']
            }
        }
    
    def identify_economic_cycles(self) -> List[Dict]:
        """Identify and classify economic cycles in the data"""
        values = self.data['Remittances (million USD)'].values
        years = self.data['Year'].values
        yoy_changes = self.processed_data['yoy_changes']
        
        cycles = []
        
        # Find peaks and troughs
        peaks, _ = find_peaks(values, height=np.mean(values), distance=3)
        troughs, _ = find_peaks(-values, height=-np.mean(values), distance=3)
        
        # Combine and sort turning points
        turning_points = []
        for peak in peaks:
            turning_points.append({'index': peak, 'type': 'peak', 'year': years[peak], 'value': values[peak]})
        for trough in troughs:
            turning_points.append({'index': trough, 'type': 'trough', 'year': years[trough], 'value': values[trough]})
        
        turning_points.sort(key=lambda x: x['year'])
        
        # Identify cycles between turning points
        for i in range(len(turning_points) - 1):
            current = turning_points[i]
            next_point = turning_points[i + 1]
            
            cycle_start = int(current['year'])
            cycle_end = int(next_point['year'])
            duration = cycle_end - cycle_start
            
            # Calculate cycle characteristics
            cycle_values = values[current['index']:next_point['index'] + 1]
            cycle_growth = (cycle_values[-1] / cycle_values[0] - 1) * 100 if len(cycle_values) > 1 else 0
            
            cycle_type = self._classify_cycle(current['type'], next_point['type'], cycle_growth, duration)
            
            cycles.append({
                'start_year': cycle_start,
                'end_year': cycle_end,
                'duration': duration,
                'start_value': float(current['value']),
                'end_value': float(next_point['value']),
                'total_growth': float(cycle_growth),
                'cycle_type': cycle_type,
                'phase': f"{current['type']}_to_{next_point['type']}",
                'economic_events': self._get_events_in_period(cycle_start, cycle_end)
            })
        
        return cycles
    
    def _classify_cycle(self, start_type: str, end_type: str, growth: float, duration: int) -> str:
        """Classify the type of economic cycle"""
        if start_type == 'trough' and end_type == 'peak':
            if growth > 50:
                return 'boom_cycle'
            elif growth > 20:
                return 'recovery_cycle'
            else:
                return 'gradual_recovery'
        elif start_type == 'peak' and end_type == 'trough':
            if growth < -20:
                return 'crisis_cycle'
            else:
                return 'correction_cycle'
        else:
            return 'transition_cycle'
    
    def _get_events_in_period(self, start_year: int, end_year: int) -> List[Dict]:
        """Get economic events within a specific period"""
        events = []
        for year in range(start_year, end_year + 1):
            if year in self.economic_events:
                for event in self.economic_events[year]:
                    events.append({**event, 'year': year})
        return events
    
    def find_similar_periods(self, target_year: int, similarity_window: int = 5) -> List[Dict]:
        """Find historical periods similar to a target period"""
        target_start = max(0, target_year - self.data['Year'].min())
        target_end = min(len(self.data), target_start + similarity_window)
        
        if target_end - target_start < 3:
            return []
        
        target_values = self.data['Remittances (million USD)'].iloc[target_start:target_end].values
        target_pattern = (target_values - np.mean(target_values)) / np.std(target_values)
        
        similar_periods = []
        years = self.data['Year'].values
        values = self.data['Remittances (million USD)'].values
        
        # Sliding window comparison
        for i in range(len(values) - similarity_window + 1):
            if abs(years[i] - target_year) < 3:  # Skip periods too close to target
                continue
                
            comparison_values = values[i:i + similarity_window]
            if len(comparison_values) < 3:
                continue
                
            comparison_pattern = (comparison_values - np.mean(comparison_values)) / np.std(comparison_values)
            
            # Calculate similarity using cosine similarity
            similarity_score = cosine_similarity(
                target_pattern.reshape(1, -1),
                comparison_pattern.reshape(1, -1)
            )[0][0]
            
            if similarity_score > 0.7:  # High similarity threshold
                similar_periods.append({
                    'period_start': int(years[i]),
                    'period_end': int(years[i + similarity_window - 1]),
                    'similarity_score': float(similarity_score),
                    'pattern_type': self._identify_pattern_type(comparison_values),
                    'economic_context': self._get_events_in_period(years[i], years[i + similarity_window - 1])
                })
        
        return sorted(similar_periods, key=lambda x: x['similarity_score'], reverse=True)
    
    def _identify_pattern_type(self, values: np.ndarray) -> str:
        """Identify the type of pattern in a value sequence"""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Calculate trend
        x = np.arange(len(values))
        correlation, _ = pearsonr(x, values)
        
        # Calculate volatility
        volatility = np.std(np.diff(values) / values[:-1])
        
        if correlation > 0.7:
            return 'strong_uptrend'
        elif correlation < -0.7:
            return 'strong_downtrend'
        elif volatility > 0.2:
            return 'high_volatility'
        elif volatility < 0.05:
            return 'stable_growth'
        else:
            return 'mixed_pattern'
    
    def extract_key_events(self) -> List[Dict]:
        """Extract and analyze key economic events and their impacts"""
        key_events = []
        values = self.data['Remittances (million USD)'].values
        years = self.data['Year'].values
        yoy_changes = self.processed_data['yoy_changes']
        
        # Analyze each event's impact
        for year, events in self.economic_events.items():
            if year < years[0] or year > years[-1]:
                continue
                
            year_idx = np.where(years == year)[0]
            if len(year_idx) == 0:
                continue
            year_idx = year_idx[0]
            
            for event in events:
                # Calculate before/after impact
                before_period = max(0, year_idx - 2)
                after_period = min(len(values) - 1, year_idx + 2)
                
                before_avg = np.mean(values[before_period:year_idx]) if year_idx > before_period else values[year_idx]
                after_avg = np.mean(values[year_idx:after_period + 1]) if after_period > year_idx else values[year_idx]
                
                impact_magnitude = (after_avg - before_avg) / before_avg * 100 if before_avg > 0 else 0
                
                # Get YoY impact if available
                yoy_impact = yoy_changes[year_idx - 1] if year_idx > 0 and year_idx <= len(yoy_changes) else None
                
                key_events.append({
                    'year': year,
                    'event': event['event'],
                    'expected_impact': event['impact'],
                    'severity': event['severity'],
                    'actual_impact_magnitude': float(impact_magnitude),
                    'yoy_growth_impact': float(yoy_impact) if yoy_impact is not None else None,
                    'impact_classification': self._classify_impact(impact_magnitude),
                    'remittance_value': float(values[year_idx]),
                    'context': self._get_event_context(year, impact_magnitude)
                })
        
        return sorted(key_events, key=lambda x: abs(x['actual_impact_magnitude']), reverse=True)
    
    def _classify_impact(self, magnitude: float) -> str:
        """Classify the impact magnitude"""
        if abs(magnitude) < 5:
            return 'minimal'
        elif abs(magnitude) < 15:
            return 'moderate'
        elif abs(magnitude) < 30:
            return 'significant'
        else:
            return 'extreme'
    
    def _get_event_context(self, year: int, impact: float) -> str:
        """Generate contextual description for event impact"""
        direction = 'positive' if impact > 0 else 'negative' if impact < 0 else 'neutral'
        magnitude = abs(impact)
        
        if magnitude < 5:
            return f"Had minimal {direction} impact on remittances"
        elif magnitude < 15:
            return f"Resulted in moderate {direction} impact ({magnitude:.1f}% change)"
        elif magnitude < 30:
            return f"Caused significant {direction} impact ({magnitude:.1f}% change)"
        else:
            return f"Triggered extreme {direction} impact ({magnitude:.1f}% change)"
    
    def analyze_seasonal_patterns(self) -> Dict:
        """Analyze seasonal and cyclical patterns"""
        # Since we have annual data, look for multi-year cycles
        values = self.data['Remittances (million USD)'].values
        years = self.data['Year'].values
        
        # Look for cyclical patterns
        cycles_analysis = {
            'business_cycles': [],
            'long_term_trends': {},
            'periodicity': None
        }
        
        # Identify potential business cycles (typically 3-7 years)
        for cycle_length in range(3, 8):
            if len(values) < cycle_length * 2:
                continue
                
            correlations = []
            for lag in range(cycle_length, len(values) - cycle_length):
                current_segment = values[lag-cycle_length:lag]
                future_segment = values[lag:lag+cycle_length]
                
                if len(current_segment) == len(future_segment):
                    corr, _ = pearsonr(current_segment, future_segment)
                    correlations.append(corr)
            
            if correlations:
                avg_correlation = np.mean(correlations)
                if avg_correlation > 0.6:
                    cycles_analysis['business_cycles'].append({
                        'cycle_length': cycle_length,
                        'correlation': float(avg_correlation),
                        'strength': 'strong' if avg_correlation > 0.8 else 'moderate'
                    })
        
        # Long-term trend analysis
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        cycles_analysis['long_term_trends'] = {
            'first_half_avg': float(np.mean(first_half)),
            'second_half_avg': float(np.mean(second_half)),
            'acceleration': float(np.mean(second_half) - np.mean(first_half)),
            'trend_change': 'accelerating' if np.mean(second_half) > np.mean(first_half) * 1.1 else 'stable'
        }
        
        return cycles_analysis
    
    def pattern_based_insights(self) -> Dict:
        """Generate comprehensive pattern-based insights"""
        insights = {
            'economic_cycles': self.identify_economic_cycles(),
            'key_events_analysis': self.extract_key_events(),
            'seasonal_patterns': self.analyze_seasonal_patterns(),
            'pattern_summary': {}
        }
        
        # Generate summary insights
        cycles = insights['economic_cycles']
        events = insights['key_events_analysis']
        
        # Most impactful events
        top_events = sorted(events, key=lambda x: abs(x['actual_impact_magnitude']), reverse=True)[:3]
        
        # Longest growth cycles
        growth_cycles = [c for c in cycles if 'recovery' in c['cycle_type'] or 'boom' in c['cycle_type']]
        longest_growth = max(growth_cycles, key=lambda x: x['duration']) if growth_cycles else None
        
        insights['pattern_summary'] = {
            'total_cycles_identified': len(cycles),
            'most_impactful_events': [e['event'] + f" ({e['year']})" for e in top_events],
            'longest_growth_period': {
                'period': f"{longest_growth['start_year']}-{longest_growth['end_year']}",
                'duration': longest_growth['duration'],
                'growth': longest_growth['total_growth']
            } if longest_growth else None,
            'pattern_characteristics': self._summarize_patterns(cycles, events)
        }
        
        return insights
    
    def _summarize_patterns(self, cycles: List[Dict], events: List[Dict]) -> Dict:
        """Summarize overall pattern characteristics"""
        # Recovery resilience
        crisis_events = [e for e in events if 'crisis' in e['event'].lower() or e['severity'] == 'extreme']
        recovery_cycles = [c for c in cycles if 'recovery' in c['cycle_type']]
        
        return {
            'resilience_score': len(recovery_cycles) / max(len(crisis_events), 1),
            'volatility_periods': len([c for c in cycles if c['duration'] < 3]),
            'stability_periods': len([c for c in cycles if c['duration'] >= 5]),
            'crisis_recovery_time': np.mean([c['duration'] for c in recovery_cycles]) if recovery_cycles else None
        }

if __name__ == "__main__":
    # Test the pattern matcher
    try:
        from data_engine import RemittanceAnalyzer
        analyzer = RemittanceAnalyzer("data/Bangladesh Remittances Dataset (19952025).csv")
        pattern_matcher = PatternMatcher(analyzer)
        
        insights = pattern_matcher.pattern_based_insights()
        print("Pattern Recognition Engine loaded successfully!")
        print(f"Economic cycles identified: {insights['pattern_summary']['total_cycles_identified']}")
        print(f"Most impactful events: {insights['pattern_summary']['most_impactful_events'][:2]}")
    except Exception as e:
        print(f"Error testing pattern matcher: {e}")
        import traceback
        traceback.print_exc()
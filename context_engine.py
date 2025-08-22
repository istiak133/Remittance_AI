"""
Context Building Engine for Rich Economic Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class ContextEngine:
    """Build rich contextual information for AI analysis"""
    
    def __init__(self, data_engine, pattern_matcher):
        """Initialize with data and pattern engines"""
        self.analyzer = data_engine
        self.pattern_matcher = pattern_matcher
        self.data = data_engine.data
        
        # Enhanced economic context database
        self.economic_context = self._build_economic_context_db()
        self.global_context = self._build_global_context()
        self.policy_context = self._build_policy_context()
        
        logger.info("Context engine initialized with comprehensive databases")
    
    def _build_economic_context_db(self) -> Dict:
        """Build comprehensive economic context database"""
        return {
            'financial_crises': {
                1997: {
                    'name': 'Asian Financial Crisis',
                    'context': 'Currency devaluations across Asia, capital flight, reduced investor confidence',
                    'bd_impact': 'Reduced remittances from Middle East, currency pressure',
                    'duration': '1997-1999',
                    'global_scope': 'Regional (Asia)',
                    'recovery_pattern': 'V-shaped recovery over 2-3 years'
                },
                2008: {
                    'name': 'Global Financial Crisis',
                    'context': 'Subprime mortgage crisis, banking sector collapse, global recession',
                    'bd_impact': 'Job losses in destination countries, reduced income of migrants',
                    'duration': '2008-2012',
                    'global_scope': 'Global',
                    'recovery_pattern': 'L-shaped recovery, slow and prolonged'
                },
                2020: {
                    'name': 'COVID-19 Pandemic',
                    'context': 'Global lockdowns, economic shutdown, supply chain disruption',
                    'bd_impact': 'Initial drop due to job losses, later surge from fiscal stimulus',
                    'duration': '2020-2022',
                    'global_scope': 'Global',
                    'recovery_pattern': 'K-shaped recovery, uneven across sectors'
                }
            },
            'growth_periods': {
                2003: {
                    'name': 'Post-Dotcom Recovery',
                    'context': 'Recovery from dot-com bubble, low interest rates, economic expansion',
                    'bd_impact': 'Increased overseas employment, stable remittance growth',
                    'characteristics': ['Gradual growth', 'Stable employment', 'Low volatility']
                },
                2010: {
                    'name': 'Post-Crisis Expansion',
                    'context': 'Recovery from 2008 crisis, quantitative easing, emerging market boom',
                    'bd_impact': 'Strong remittance growth, new migration opportunities',
                    'characteristics': ['Rapid recovery', 'High growth rates', 'Increased volatility']
                }
            },
            'external_shocks': {
                2001: {
                    'name': '9/11 Attacks',
                    'context': 'Security concerns, immigration restrictions, economic uncertainty',
                    'bd_impact': 'Temporary reduction in migration, increased transaction costs',
                    'type': 'Security shock'
                },
                2014: {
                    'name': 'Oil Price Collapse',
                    'context': 'Oil prices fell from $100+ to $30, affecting oil-dependent economies',
                    'bd_impact': 'Reduced incomes in Gulf countries, lower remittances from Middle East',
                    'type': 'Commodity shock'
                },
                2016: {
                    'name': 'Brexit Vote',
                    'context': 'UK votes to leave EU, currency volatility, economic uncertainty',
                    'bd_impact': 'Pound devaluation, reduced remittance values in USD terms',
                    'type': 'Political shock'
                }
            }
        }
    
    def _build_global_context(self) -> Dict:
        """Build global economic context"""
        return {
            'migration_trends': {
                1990: 'Beginning of large-scale labor migration to Gulf countries',
                2000: 'Diversification to Southeast Asia and Europe',
                2010: 'Expansion to North America and skilled migration increase',
                2020: 'Digital remittances adoption, COVID-19 migration disruptions'
            },
            'technology_evolution': {
                1995: 'Traditional banking, high transaction costs',
                2005: 'Money transfer operators expansion',
                2010: 'Mobile money introduction',
                2015: 'Digital remittances mainstream adoption',
                2020: 'Fintech solutions, reduced costs'
            },
            'policy_environment': {
                2000: 'Remittance promotion policies introduced',
                2005: 'Incentive schemes for formal channels',
                2010: 'Mobile financial services regulations',
                2015: 'Financial inclusion initiatives',
                2020: 'Digital payment infrastructure expansion'
            }
        }
    
    def _build_policy_context(self) -> Dict:
        """Build Bangladesh-specific policy context"""
        return {
            'remittance_policies': {
                2002: 'Wage Earners Development Bond introduced',
                2009: '2% incentive on remittances through banking channels',
                2019: 'Mobile financial services regulations updated',
                2020: 'Emergency remittance measures during COVID-19'
            },
            'economic_policies': {
                1990: 'Economic liberalization begins',
                2000: 'Export-oriented growth strategy',
                2010: 'Vision 2021 development plan',
                2020: 'Vision 2041 and digital transformation'
            },
            'exchange_rate_policies': {
                1999: 'Flexible exchange rate regime adopted',
                2003: 'Managed float system',
                2012: 'Greater exchange rate flexibility',
                2022: 'Exchange rate adjustments during global inflation'
            }
        }
    
    def contextualize_period(self, start_year: int, end_year: int) -> Dict:
        """Generate rich contextual analysis for a specific period"""
        period_context = {
            'period': f"{start_year}-{end_year}",
            'duration': end_year - start_year + 1,
            'economic_events': [],
            'global_trends': [],
            'policy_changes': [],
            'pattern_analysis': {},
            'comparative_context': {}
        }
        
        # Extract relevant events
        for year in range(start_year, end_year + 1):
            # Financial crises
            for crisis_year, crisis_info in self.economic_context['financial_crises'].items():
                if self._year_in_range(year, crisis_year, crisis_info.get('duration', str(crisis_year))):
                    period_context['economic_events'].append({
                        'year': crisis_year,
                        'type': 'financial_crisis',
                        'event': crisis_info['name'],
                        'impact': crisis_info['bd_impact'],
                        'context': crisis_info['context']
                    })
            
            # External shocks
            if year in self.economic_context['external_shocks']:
                shock = self.economic_context['external_shocks'][year]
                period_context['economic_events'].append({
                    'year': year,
                    'type': 'external_shock',
                    'event': shock['name'],
                    'impact': shock['bd_impact'],
                    'context': shock['context']
                })
            
            # Policy changes
            for policy_type, policies in self.policy_context.items():
                if year in policies:
                    period_context['policy_changes'].append({
                        'year': year,
                        'type': policy_type,
                        'policy': policies[year]
                    })
        
        # Get pattern analysis for the period
        try:
            period_data = self.analyzer.get_period_analysis(start_year, end_year)
            period_context['pattern_analysis'] = period_data
        except:
            pass
        
        # Find similar historical periods
        try:
            similar_periods = self.pattern_matcher.find_similar_periods(
                start_year + (end_year - start_year) // 2, 
                end_year - start_year + 1
            )
            period_context['comparative_context'] = {
                'similar_periods': similar_periods[:3],  # Top 3 similar periods
                'uniqueness_score': 1.0 - (similar_periods[0]['similarity_score'] if similar_periods else 0)
            }
        except:
            pass
        
        return period_context
    
    def _year_in_range(self, year: int, event_year: int, duration_str: str) -> bool:
        """Check if year falls within event duration"""
        try:
            if '-' in duration_str:
                start, end = duration_str.split('-')
                return int(start) <= year <= int(end)
            else:
                return year == event_year
        except:
            return year == event_year
    
    def build_query_context(self, user_query: str, relevant_years: List[int] = None) -> Dict:
        """Build context for a specific user query"""
        context = {
            'query': user_query,
            'query_type': self._classify_query(user_query),
            'relevant_data': {},
            'historical_context': {},
            'comparative_insights': {},
            'economic_narrative': ""
        }
        
        # Extract years mentioned in query or use relevant years
        if relevant_years is None:
            relevant_years = self._extract_years_from_query(user_query)
        
        # Build context for relevant years
        if relevant_years:
            year_range = (min(relevant_years), max(relevant_years))
            context['historical_context'] = self.contextualize_period(year_range[0], year_range[1])
            
            # Get specific data for these years
            context['relevant_data'] = self._get_data_for_years(relevant_years)
        
        # Add comparative context based on query type
        if context['query_type'] in ['comparison', 'trend_analysis']:
            context['comparative_insights'] = self._build_comparative_context(user_query, relevant_years)
        
        # Generate narrative thread
        context['economic_narrative'] = self._generate_narrative_thread(context)
        
        return context
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of user query"""
        query_lower = query.lower()
        
        comparison_keywords = ['compare', 'vs', 'versus', 'difference', 'better', 'worse']
        trend_keywords = ['trend', 'pattern', 'growth', 'decline', 'increase', 'decrease']
        causal_keywords = ['why', 'because', 'reason', 'cause', 'due to', 'impact of']
        forecast_keywords = ['predict', 'forecast', 'future', 'next', 'will', 'expect']
        anomaly_keywords = ['unusual', 'strange', 'anomaly', 'outlier', 'abnormal']
        
        if any(keyword in query_lower for keyword in comparison_keywords):
            return 'comparison'
        elif any(keyword in query_lower for keyword in causal_keywords):
            return 'causal_analysis'
        elif any(keyword in query_lower for keyword in forecast_keywords):
            return 'forecasting'
        elif any(keyword in query_lower for keyword in anomaly_keywords):
            return 'anomaly_detection'
        elif any(keyword in query_lower for keyword in trend_keywords):
            return 'trend_analysis'
        else:
            return 'general_inquiry'
    
    def _extract_years_from_query(self, query: str) -> List[int]:
        """Extract years mentioned in the query"""
        import re
        years = []
        
        # Find 4-digit years
        year_matches = re.findall(r'\b(19|20)\d{2}\b', query)
        for match in year_matches:
            year = int(match)
            if 1995 <= year <= 2025:  # Within our data range
                years.append(year)
        
        # Find periods like "2008 crisis", "COVID", etc.
        event_mappings = {
            'covid': [2020, 2021, 2022],
            'pandemic': [2020, 2021, 2022],
            '2008 crisis': [2008, 2009, 2010],
            'financial crisis': [2008, 2009, 2010],
            'asian crisis': [1997, 1998, 1999],
            'brexit': [2016, 2017],
            '9/11': [2001, 2002]
        }
        
        query_lower = query.lower()
        for event, event_years in event_mappings.items():
            if event in query_lower:
                years.extend(event_years)
        
        return sorted(list(set(years)))  # Remove duplicates and sort
    
    def _get_data_for_years(self, years: List[int]) -> Dict:
        """Get relevant data for specific years"""
        relevant_data = {}
        
        for year in years:
            year_data = self.data[self.data['Year'] == year]
            if not year_data.empty:
                relevant_data[year] = {
                    'remittances_usd': float(year_data['Remittances (million USD)'].iloc[0]),
                    'remittances_bdt': float(year_data['Remittances (billion BDT)'].iloc[0]) if 'Remittances (billion BDT)' in year_data.columns else None,
                    'yoy_change': float(year_data['YoY Change (%)'].iloc[0]) if 'YoY Change (%)' in year_data.columns else None
                }
        
        return relevant_data
    
    def _build_comparative_context(self, query: str, years: List[int]) -> Dict:
        """Build comparative context for analysis"""
        if not years or len(years) < 2:
            return {}
        
        comparative_context = {
            'periods_compared': [],
            'benchmarks': {},
            'relative_performance': {}
        }
        
        # If comparing specific periods
        if len(years) >= 2:
            period1_data = self._get_data_for_years(years[:len(years)//2])
            period2_data = self._get_data_for_years(years[len(years)//2:])
            
            comparative_context['periods_compared'] = [
                {'period': f"{min(years[:len(years)//2])}-{max(years[:len(years)//2])}", 'data': period1_data},
                {'period': f"{min(years[len(years)//2:])}-{max(years[len(years)//2:])}", 'data': period2_data}
            ]
        
        # Add benchmark comparisons
        comparative_context['benchmarks'] = {
            'historical_average': float(self.data['Remittances (million USD)'].mean()),
            'pre_2008_average': float(self.data[self.data['Year'] < 2008]['Remittances (million USD)'].mean()),
            'post_2008_average': float(self.data[self.data['Year'] >= 2008]['Remittances (million USD)'].mean()),
            'last_decade_average': float(self.data[self.data['Year'] >= 2014]['Remittances (million USD)'].mean())
        }
        
        return comparative_context
    
    def _generate_narrative_thread(self, context: Dict) -> str:
        """Generate a coherent narrative thread for the context"""
        narrative_parts = []
        
        # Add period context
        if 'historical_context' in context and context['historical_context']:
            hist_context = context['historical_context']
            period = hist_context.get('period', '')
            
            if hist_context.get('economic_events'):
                events = [event['event'] for event in hist_context['economic_events'][:2]]
                narrative_parts.append(f"During {period}, key events included {' and '.join(events)}")
            
            if hist_context.get('pattern_analysis') and 'trend' in hist_context['pattern_analysis']:
                trend = hist_context['pattern_analysis']['trend']
                narrative_parts.append(f"The overall trend was {trend}")
        
        # Add comparative context
        if context.get('comparative_insights') and context['comparative_insights'].get('benchmarks'):
            benchmarks = context['comparative_insights']['benchmarks']
            narrative_parts.append(f"Historical average remittances: ${benchmarks['historical_average']:.1f}M")
        
        return '. '.join(narrative_parts) if narrative_parts else "Limited historical context available for this query."
    
    def get_comprehensive_context(self, query: str = None, years: List[int] = None, include_patterns: bool = True) -> Dict:
        """Get comprehensive context for AI processing"""
        context = {
            'data_summary': self._get_data_summary(),
            'economic_timeline': self._get_economic_timeline(),
            'pattern_insights': {},
            'policy_environment': self.policy_context,
            'global_trends': self.global_context
        }
        
        # Add query-specific context if provided
        if query:
            context['query_context'] = self.build_query_context(query, years)
        
        # Add pattern insights if requested
        if include_patterns:
            try:
                context['pattern_insights'] = self.pattern_matcher.pattern_based_insights()
            except Exception as e:
                logger.warning(f"Could not generate pattern insights: {e}")
                context['pattern_insights'] = {}
        
        return context
    
    def _get_data_summary(self) -> Dict:
        """Get high-level data summary"""
        data = self.data['Remittances (million USD)']
        return {
            'time_span': f"{self.data['Year'].min()}-{self.data['Year'].max()}",
            'total_years': len(self.data),
            'value_range': f"${data.min():.1f}M - ${data.max():.1f}M",
            'average_value': f"${data.mean():.1f}M",
            'total_cumulative': f"${data.sum():.1f}M",
            'compound_growth': f"{((data.iloc[-1]/data.iloc[0])**(1/(len(data)-1)) - 1)*100:.2f}%"
        }
    
    def _get_economic_timeline(self) -> List[Dict]:
        """Get chronological timeline of major events"""
        timeline = []
        
        # Combine all events with years
        all_events = []
        
        # Financial crises
        for year, crisis in self.economic_context['financial_crises'].items():
            all_events.append({
                'year': year,
                'type': 'financial_crisis',
                'event': crisis['name'],
                'impact': crisis['bd_impact'],
                'severity': 'high'
            })
        
        # External shocks
        for year, shock in self.economic_context['external_shocks'].items():
            all_events.append({
                'year': year,
                'type': 'external_shock',
                'event': shock['name'],
                'impact': shock['bd_impact'],
                'severity': 'medium'
            })
        
        # Policy changes (major ones only)
        major_policies = {
            2002: 'Wage Earners Development Bond introduced',
            2009: '2% incentive on remittances introduced',
            2019: 'Mobile financial services regulations updated'
        }
        
        for year, policy in major_policies.items():
            all_events.append({
                'year': year,
                'type': 'policy_change',
                'event': policy,
                'impact': 'Positive impact on formal remittance channels',
                'severity': 'medium'
            })
        
        return sorted(all_events, key=lambda x: x['year'])
    
    def contextualize_anomaly(self, year: int, anomaly_type: str, magnitude: float) -> Dict:
        """Provide rich context for detected anomalies"""
        context = {
            'year': year,
            'anomaly_type': anomaly_type,
            'magnitude': magnitude,
            'possible_explanations': [],
            'historical_precedents': [],
            'economic_environment': {}
        }
        
        # Get economic environment for that year
        year_context = self.contextualize_period(year, year)
        context['economic_environment'] = year_context
        
        # Find possible explanations based on events
        for event in year_context.get('economic_events', []):
            if anomaly_type == 'high' and 'positive' in event.get('impact', '').lower():
                context['possible_explanations'].append(f"{event['event']}: {event['impact']}")
            elif anomaly_type == 'low' and 'negative' in event.get('impact', '').lower():
                context['possible_explanations'].append(f"{event['event']}: {event['impact']}")
        
        # Find historical precedents
        try:
            similar_periods = self.pattern_matcher.find_similar_periods(year, 1)
            context['historical_precedents'] = similar_periods[:2]  # Top 2 similar situations
        except:
            pass
        
        return context
    
    def generate_executive_context(self, focus_area: str = 'overview') -> Dict:
        """Generate executive-level context for reports"""
        context = {
            'focus_area': focus_area,
            'key_highlights': [],
            'major_trends': [],
            'risk_factors': [],
            'opportunities': []
        }
        
        # Key highlights based on data
        data = self.data['Remittances (million USD)']
        latest_year = self.data['Year'].iloc[-1]
        latest_value = data.iloc[-1]
        
        context['key_highlights'] = [
            f"Remittances reached ${latest_value:.1f}M in {latest_year}",
            f"Compound annual growth of {((data.iloc[-1]/data.iloc[0])**(1/(len(data)-1)) - 1)*100:.1f}% over {len(data)} years",
            f"Total cumulative remittances: ${data.sum():.1f}M",
            f"Peak remittances: ${data.max():.1f}M in {self.data['Year'].iloc[data.argmax()]}"
        ]
        
        # Major trends
        recent_growth = ((data.iloc[-3:].mean() / data.iloc[-6:-3].mean()) - 1) * 100
        context['major_trends'] = [
            f"Recent 3-year average growth: {recent_growth:.1f}%",
            "Digitalization driving formal channel adoption",
            "Diversification of destination countries",
            "Resilience demonstrated during global crises"
        ]
        
        # Risk factors and opportunities based on patterns
        context['risk_factors'] = [
            "Global economic uncertainties",
            "Destination country policy changes",
            "Currency fluctuation impacts",
            "Technological disruption challenges"
        ]
        
        context['opportunities'] = [
            "Digital financial services expansion",
            "Cost reduction through technology",
            "Financial inclusion improvements",
            "Investment channel development"
        ]
        
        return context

if __name__ == "__main__":
    # Test the context engine
    import sys
    sys.path.append('.')
    from data_engine import RemittanceAnalyzer
    from pattern_engine import PatternMatcher
    
    analyzer = RemittanceAnalyzer("../data/Bangladesh Remittances Dataset (19952025).csv")
    pattern_matcher = PatternMatcher(analyzer)
    context_engine = ContextEngine(analyzer, pattern_matcher)
    
    # Test contextualization
    covid_context = context_engine.contextualize_period(2019, 2021)
    print("Context Engine loaded successfully!")
    print(f"COVID period events: {len(covid_context['economic_events'])}")
    print(f"Policy changes identified: {len(covid_context['policy_changes'])}")
"""
Integration test for Day 1 components
"""

import sys
sys.path.append('src')

from data_engine import RemittanceAnalyzer
from pattern_engine import PatternMatcher
from context_engine import ContextEngine
import json

def test_integration():
    print("🔧 Running Day 1 Integration Test...")
    print("="*50)
    
    # Initialize all components
    try:
        print("🏗️ Initializing components...")
        analyzer = RemittanceAnalyzer("data/Bangladesh Remittances Dataset (19952025).csv")
        pattern_matcher = PatternMatcher(analyzer)
        context_engine = ContextEngine(analyzer, pattern_matcher)
        print("✅ All components initialized successfully!")
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return False
    
    print("\n🧠 Testing Intelligence Pipeline...")
    
    # Test 1: Complex Query Processing
    try:
        query = "Analyze the impact of COVID-19 on Bangladesh remittances compared to the 2008 financial crisis"
        
        # Step 1: Extract insights from data
        covid_period = analyzer.get_period_analysis(2019, 2021)
        crisis_period = analyzer.get_period_analysis(2008, 2010)
        
        # Step 2: Find patterns
        similar_periods = pattern_matcher.find_similar_periods(2020, 3)
        key_events = pattern_matcher.extract_key_events()
        
        # Step 3: Build context
        comprehensive_context = context_engine.get_comprehensive_context(query)
        
        print("✅ Complex query processing pipeline working!")
        print(f"   📊 COVID period average: ${covid_period['average_annual']:.1f}M")
        print(f"   📊 2008 crisis period average: ${crisis_period['average_annual']:.1f}M")
        print(f"   🔍 Similar periods found: {len(similar_periods)}")
        
    except Exception as e:
        print(f"❌ Complex query processing failed: {e}")
        return False
    
    # Test 2: Anomaly Analysis with Context
    try:
        anomalies = analyzer.detect_anomalies('iqr')
        if anomalies:
            major_anomaly = max(anomalies, key=lambda x: abs(x['value'] - analyzer.data['Remittances (million USD)'].mean()))
            anomaly_context = context_engine.contextualize_anomaly(
                major_anomaly['year'], 
                major_anomaly['type'],
                major_anomaly['value']
            )
            print(f"✅ Anomaly analysis with context working!")
            print(f"   🚨 Major anomaly: {major_anomaly['year']} ({major_anomaly['type']})")
            print(f"   💡 Explanations found: {len(anomaly_context['possible_explanations'])}")
        else:
            print("ℹ️ No anomalies found to test contextualization")
        
    except Exception as e:
        print(f"❌ Anomaly analysis failed: {e}")
        return False
    
    # Test 3: Executive Intelligence
    try:
        exec_context = context_engine.generate_executive_context()
        pattern_insights = pattern_matcher.pattern_based_insights()
        
        print("✅ Executive intelligence generation working!")
        print(f"   📈 Key highlights: {len(exec_context['key_highlights'])}")
        print(f"   🔄 Economic cycles identified: {pattern_insights['pattern_summary']['total_cycles_identified']}")
        
    except Exception as e:
        print(f"❌ Executive intelligence failed: {e}")
        return False
    
    # Test 4: Data Quality and Coverage
    try:
        data_coverage = {
            'years_covered': len(analyzer.data),
            'data_completeness': analyzer.data['Remittances (million USD)'].notna().sum() / len(analyzer.data),
            'anomalies_detected': len(analyzer.detect_anomalies('iqr')),
            'patterns_identified': len(pattern_matcher.identify_economic_cycles()),
            'events_catalogued': len(context_engine._get_economic_timeline())
        }
        
        print("✅ Data quality assessment completed!")
        print(f"   📊 Years covered: {data_coverage['years_covered']}")
        print(f"   ✨ Data completeness: {data_coverage['data_completeness']:.1%}")
        print(f"   🔍 Total anomalies: {data_coverage['anomalies_detected']}")
        print(f"   🔄 Pattern cycles: {data_coverage['patterns_identified']}")
        print(f"   📅 Historical events: {data_coverage['events_catalogued']}")
        
    except Exception as e:
        print(f"❌ Data quality assessment failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("🎉 DAY 1 INTEGRATION TEST PASSED!")
    print("🚀 Ready for Day 2: AI Integration")
    print("="*50)
    
    return True

if __name__ == "__main__":
    test_integration()

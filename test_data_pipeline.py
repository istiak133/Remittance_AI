"""
Test script for the Data Pipeline
"""

import sys
sys.path.append('src')

from data_pipeline import DataPipeline
import os

def test_data_pipeline():
    print("🔧 Testing Data Pipeline...")
    print("="*50)
    
    # Initialize pipeline
    try:
        pipeline = DataPipeline("data/remittance_core.db")
        print("✅ Data pipeline initialized successfully!")
        
        # Clear database for fresh test
        pipeline.clear_database()
        print("🧹 Database cleared for fresh testing")
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False
    
    # Test 1: File Upload and Processing
    try:
        print("\n📁 Testing file upload and processing...")
        result = pipeline.upload_file("data/Bangladesh Remittances Dataset (19952025).csv")
        
        if result['success']:
            print(f"✅ File upload successful!")
            print(f"   📊 Records processed: {result['records_processed']}")
            print(f"   📊 Records stored: {result['records_stored']}")
            print(f"   🎯 Quality score: {result['quality_score']:.3f}")
            print(f"   🔐 Data hash: {result['data_hash'][:16]}...")
        else:
            print(f"❌ File upload failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ File upload test failed: {e}")
        return False
    
    # Test 2: Data Summary
    try:
        print("\n📊 Testing data summary...")
        summary = pipeline.get_data_summary()
        
        if summary:
            print(f"✅ Data summary retrieved!")
            print(f"   📊 Total records: {summary['total_records']}")
            print(f"   📅 Year range: {summary['year_range']}")
            print(f"   🎯 Average quality: {summary['average_quality_score']:.3f}")
            print(f"   📁 Database path: {summary['database_path']}")
            print(f"   🔗 Data sources: {len(summary['data_sources'])}")
            
            for source in summary['data_sources']:
                print(f"      - {source['name']}: {source['records']} records")
        else:
            print("❌ Failed to retrieve data summary")
            return False
            
    except Exception as e:
        print(f"❌ Data summary test failed: {e}")
        return False
    
    # Test 3: Data Export
    try:
        print("\n📤 Testing data export...")
        export_path = pipeline.export_data('csv')
        
        if os.path.exists(export_path):
            print(f"✅ Data export successful!")
            print(f"   📁 Export path: {export_path}")
            print(f"   📏 File size: {os.path.getsize(export_path)} bytes")
        else:
            print("❌ Data export failed")
            return False
            
    except Exception as e:
        print(f"❌ Data export test failed: {e}")
        return False
    
    # Test 4: Duplicate Upload Prevention
    try:
        print("\n🔄 Testing duplicate upload prevention...")
        duplicate_result = pipeline.upload_file("data/Bangladesh Remittances Dataset (19952025).csv")
        
        if duplicate_result['success']:
            print(f"✅ Duplicate handling working!")
            print(f"   📊 Message: {duplicate_result['message']}")
        else:
            print(f"❌ Duplicate handling failed: {duplicate_result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Duplicate upload test failed: {e}")
        return False
    
    # Test 5: Database Structure
    try:
        print("\n🗄️ Testing database structure...")
        
        # Check if tables exist
        cursor = pipeline.db_connection.cursor()
        
        # Check remittance_data table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='remittance_data'")
        if cursor.fetchone():
            print("✅ remittance_data table exists")
        else:
            print("❌ remittance_data table missing")
            return False
        
        # Check data_quality_log table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data_quality_log'")
        if cursor.fetchone():
            print("✅ data_quality_log table exists")
        else:
            print("❌ data_quality_log table missing")
            return False
        
        # Check data_sources table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data_sources'")
        if cursor.fetchone():
            print("✅ data_sources table exists")
        else:
            print("❌ data_sources table missing")
            return False
        
        # Check record counts
        cursor.execute("SELECT COUNT(*) FROM remittance_data")
        data_count = cursor.fetchone()[0]
        print(f"✅ remittance_data: {data_count} records")
        
        cursor.execute("SELECT COUNT(*) FROM data_quality_log")
        log_count = cursor.fetchone()[0]
        print(f"✅ data_quality_log: {log_count} records")
        
        cursor.execute("SELECT COUNT(*) FROM data_sources")
        source_count = cursor.fetchone()[0]
        print(f"✅ data_sources: {source_count} records")
        
    except Exception as e:
        print(f"❌ Database structure test failed: {e}")
        return False
    
    # Cleanup
    try:
        pipeline.close()
        print("\n🔒 Database connection closed")
    except Exception as e:
        print(f"⚠️ Warning: Could not close database connection: {e}")
    
    print("\n" + "="*50)
    print("🎉 DATA PIPELINE TEST PASSED!")
    print("🚀 Ready for next feature: Chat Assistant")
    print("="*50)
    
    return True

if __name__ == "__main__":
    test_data_pipeline()


from src.rag_pipeline import RemittanceRAGPipeline
import sys

def main():
    # Process command line argument if provided
    if len(sys.argv) > 1:
        # Join all arguments as a single query
        query = " ".join(sys.argv[1:])
        print(f"Processing query: {query}")
        process_query(query)
        return
        
    # Interactive mode if no arguments provided
    # Initialize the pipeline
    print("Initializing Bangladesh Remittance RAG Pipeline...")
    rag = RemittanceRAGPipeline('data/Bangladesh Remittances Dataset (19952025).csv')
    print("\n" + "-"*70)
    print("BANGLADESH REMITTANCE QUERY SYSTEM")
    print("-"*70)
    print("Type your question and press Enter (type 'exit' to quit)")
    print("\nExample questions:")
    print("- What was the remittance in 2020?")
    print("- How much did remittance grow in 2019?")
    print("- Show me the data for 2022")
    print("- Compare remittance between 2021 and 2022")
    print("-"*70)
    
    while True:
        try:
            query = input("\nQ: ")
            if query.strip().lower() == 'exit':
                print("Exiting.")
                break
            
            process_query(query, rag)
            
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def process_query(query, rag=None):
    """Process a query and display results"""
    # Initialize RAG if not provided
    if rag is None:
        rag = RemittanceRAGPipeline('data/Bangladesh Remittances Dataset (19952025).csv')
    
    try:
        # Retrieve and display relevant data
        results = rag.retrieve(query, top_k=3)
        
        # Generate and display answer
        answer = rag.generate(query, top_k=3)
        print("\n" + "="*50)
        print("ANSWER:")
        print("="*50)
        print(answer)
        print("="*50)
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()

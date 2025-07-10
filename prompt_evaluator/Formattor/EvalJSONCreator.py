#!/usr/bin/env python3
"""
Azure AI Toolkit Evaluation JSONL Creator - Simplified Version
Creates clean evaluation files with only query, response, and ground_truth fields
"""

import json
import re
from typing import List, Dict, Any
from datetime import datetime

class EvaluationJSONLCreator:
    def __init__(self):
        pass
        
    def load_agent_outputs(self, file_path: str) -> List[Dict[str, Any]]:
        """Load agent outputs from outputPrompt01.jsonl"""
        outputs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            
                            # Extract the actual response
                            response = data.get('response', '').strip()
                            
                            outputs.append({
                                'response': response,
                                'line_number': line_num
                            })
                            
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse line {line_num}: {e}")
                            continue
                            
        except FileNotFoundError:
            print(f"Error: Could not find file {file_path}")
            return []
        
        print(f"✅ Loaded {len(outputs)} agent outputs")
        return outputs

    def load_ground_truth(self, file_path: str) -> List[Dict[str, Any]]:
        """Load ground truth from ground_truth.jsonl"""
        ground_truths = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            ground_truths.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse ground truth line {line_num}: {e}")
                            continue
                            
        except FileNotFoundError:
            print(f"Error: Could not find file {file_path}")
            return []
        
        print(f"✅ Loaded {len(ground_truths)} ground truth entries")
        return ground_truths

    def find_ev_charging_ground_truth(self, ground_truths: List[Dict]) -> Dict:
        """Find the EV charging ground truth entry"""
        for gt in ground_truths:
            if "electric vehicle" in gt.get('query', '').lower():
                return gt
        
        # If not found, return the first one as fallback
        return ground_truths[0] if ground_truths else {}

    def create_evaluation_entries(self, agent_outputs: List[Dict], 
                                ev_ground_truth: Dict) -> List[Dict]:
        """Create simplified evaluation entries with only query, response, and ground_truth"""
        evaluation_entries = []
        
        query = ev_ground_truth.get('query', 'When is the best time to charge my electric vehicle?')
        ground_truth_response = ev_ground_truth.get('ground_truth', '')
        
        for i, output in enumerate(agent_outputs):
            entry = {
                "query": query,
                "response": output['response'],
                "ground_truth": ground_truth_response
            }
            evaluation_entries.append(entry)
            
        return evaluation_entries

    def save_evaluation_file(self, entries: List[Dict], output_file: str):
        """Save evaluation entries to JSONL file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"✅ Saved {len(entries)} evaluation entries to {output_file}")
            
            # Print summary
            print(f"📊 Evaluation Summary:")
            print(f"   - Query: {entries[0]['query'] if entries else 'N/A'}")
            print(f"   - Number of agent responses: {len(entries)}")
            print(f"   - Format: Simplified (query, response, ground_truth only)")
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")

    def analyze_responses(self, agent_outputs: List[Dict]):
        """Analyze the consistency of agent responses"""
        if not agent_outputs:
            return
            
        print(f"\n🔍 Response Analysis:")
        print(f"Total responses: {len(agent_outputs)}")
        
        # Check for identical responses
        responses = [output['response'] for output in agent_outputs]
        unique_responses = list(set(responses))
        
        print(f"Unique responses: {len(unique_responses)}")
        
        if len(unique_responses) == 1:
            print("✅ All responses are identical (perfect consistency)")
        else:
            print("⚠️  Responses vary - analyzing differences:")
            for i, response in enumerate(unique_responses, 1):
                count = responses.count(response)
                print(f"   Variant {i} (appears {count}x): {response[:100]}...")

    def show_sample_evaluation(self, entries: List[Dict]):
        """Show a sample evaluation entry"""
        if not entries:
            return
            
        print(f"\n📋 Sample Evaluation Entry:")
        sample = entries[0]
        print(f"Query: {sample['query']}")
        print(f"Response: {sample['response'][:100]}...")
        print(f"Ground Truth: {sample['ground_truth'][:100]}...")

    def create_evaluation_file(self):
        """Main method to create the evaluation file"""
        
        print("🚀 Creating Azure AI Toolkit Evaluation File...")
        print("=" * 60)
        
        # Load the source files
        agent_outputs = self.load_agent_outputs('outputPrompt01.jsonl')
        ground_truths = self.load_ground_truth('ground_truth.jsonl')
        
        if not agent_outputs:
            print("❌ No agent outputs found. Please check outputPrompt01.jsonl")
            return
            
        if not ground_truths:
            print("❌ No ground truth found. Please check ground_truth.jsonl")
            return
        
        # Find the EV charging ground truth
        ev_ground_truth = self.find_ev_charging_ground_truth(ground_truths)
        if not ev_ground_truth:
            print("❌ Could not find EV charging ground truth")
            return
            
        print(f"🎯 Found EV charging ground truth: {ev_ground_truth['query']}")
        
        # Analyze responses
        self.analyze_responses(agent_outputs)
        
        # Create evaluation entries
        evaluation_entries = self.create_evaluation_entries(agent_outputs, ev_ground_truth)
        
        # Save evaluation file
        output_file = 'azure_evaluation.jsonl'
        self.save_evaluation_file(evaluation_entries, output_file)
        
        # Show sample
        self.show_sample_evaluation(evaluation_entries)
        
        print(f"\n🎯 File ready for Azure AI Toolkit evaluation!")
        print(f"📁 Use '{output_file}' for your evaluation")
        
        print("\n📋 Next Steps:")
        print("1. Import 'azure_evaluation.jsonl' into Azure AI Toolkit")
        print("2. Configure evaluators: F1 Score, Coherence, Groundedness, Relevance")
        print("3. Run evaluation to measure response quality")
        print("4. Demonstrate consistency across multiple agent runs")

def main():
    """Main function to create evaluation files"""
    creator = EvaluationJSONLCreator()
    creator.create_evaluation_file()

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import Counter, defaultdict
import os
import re

class PartRecommendationSystem:
    """
    A recommendation system for AutoSure Insurance that suggests associated damaged parts
    based on historical claim data to assist surveyors during damage assessment.
    """
    
    def __init__(self, data_file, min_support=10, top_parts=500):
        """
        Initialize the recommendation system with the surveyor dataset.
        
        Args:
            data_file (str): Path to the surveyor data CSV file
            min_support (int): Minimum number of co-occurrences to consider valid association
            top_parts (int): Number of most common parts to include in analysis
        """
        print(f"Initializing Part Recommendation System...")
        self.min_support = min_support
        self.top_parts = top_parts
        
        # Load and preprocess data
        self.data = pd.read_csv(data_file, low_memory=False)
        self._preprocess_data()
        
        # Build associations using a memory-efficient approach
        self._build_associations()
        
        print(f"Recommendation system initialized with {len(self.part_list)} parts.")
        print(f"Built association rules with minimum support of {min_support} co-occurrences.")
    
    def _preprocess_data(self):
        """Preprocess the surveyor data for recommendation analysis."""
        print("Preprocessing data...")
        # Count part frequencies
        part_counts = self.data['TXT_PARTS_NAME'].value_counts()
        
        # Select top parts to analyze
        self.part_list = part_counts.head(self.top_parts).index.tolist()
        print(f"Selected top {len(self.part_list)} parts for analysis.")
        
        # Filter data to only include top parts
        self.filtered_data = self.data[self.data['TXT_PARTS_NAME'].isin(self.part_list)]
        
        # Create a mapping of reference numbers to parts
        self.reference_to_parts = defaultdict(list)
        for ref, part in zip(self.filtered_data['REFERENCE_NUM'], self.filtered_data['TXT_PARTS_NAME']):
            self.reference_to_parts[ref].append(part)
        
        # Create a mapping of parts to frequencies
        self.part_frequencies = Counter()
        for parts_list in self.reference_to_parts.values():
            self.part_frequencies.update(parts_list)
    
    def _build_associations(self):
        """Build association rules directly without creating a full co-occurrence matrix."""
        print("Building part associations...")
        
        # Initialize counter for co-occurrences
        cooccurrence_counts = defaultdict(Counter)
        
        # Count co-occurrences
        for parts_list in self.reference_to_parts.values():
            # Only process unique parts in this claim
            unique_parts = set(parts_list)
            for part1 in unique_parts:
                for part2 in unique_parts:
                    if part1 != part2:
                        cooccurrence_counts[part1][part2] += 1
        
        # Build association rules
        self.association_rules = {}
        for part1, associated_parts in cooccurrence_counts.items():
            self.association_rules[part1] = {}
            for part2, count in associated_parts.items():
                if count >= self.min_support:
                    # Calculate confidence: P(part2|part1)
                    confidence = count / self.part_frequencies[part1]
                    
                    # Store rule with combined score
                    self.association_rules[part1][part2] = {
                        'support': count,
                        'confidence': confidence,
                        'score': count * confidence  # Combined score
                    }
        
        print(f"Built association rules for {len(self.association_rules)} parts.")
    
    def get_recommendations(self, selected_parts, top_n=5):
        """
        Get recommended parts based on parts already selected by the surveyor.
        
        Args:
            selected_parts (list): List of part names already selected by the surveyor
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of tuples (part_name, score) for recommended parts
        """
        if not selected_parts:
            # If no parts selected, recommend most common parts
            return [(part, count) for part, count in 
                   self.part_frequencies.most_common(top_n)]
        
        # Calculate aggregated scores for all potential recommendations
        scores = Counter()
        
        for selected_part in selected_parts:
            if selected_part in self.association_rules:
                # Get all associated parts and their scores
                for candidate, rule in self.association_rules[selected_part].items():
                    if candidate not in selected_parts:  # Don't recommend already selected parts
                        scores[candidate] += rule['score']
        
        # Sort by score and return top N
        return scores.most_common(top_n)
    
    def visualize_associations(self, part_name, top_n=10, output_dir='analysis_figures'):
        """
        Visualize associations for a specific part.
        
        Args:
            part_name (str): The part to visualize associations for
            top_n (int): Number of top associations to visualize
            output_dir (str): Directory to save visualization
        """
        if part_name not in self.association_rules:
            print(f"Part '{part_name}' not found in association rules.")
            return
        
        # Get top associated parts
        associations = self.association_rules[part_name]
        top_associations = sorted(
            [(part, rule['score']) for part, rule in associations.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Create DataFrame for plotting
        assoc_df = pd.DataFrame(top_associations, columns=['Associated Part', 'Score'])
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Score', y='Associated Part', data=assoc_df,
                   palette='viridis', hue='Associated Part', legend=False)
        
        plt.title(f'Top {top_n} Parts Associated with "{part_name}"', fontsize=16)
        plt.xlabel('Association Score', fontsize=12)
        plt.ylabel('Associated Part', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        safe_name = re.sub(r'[\\/*?:"<>|]', '_', part_name)
        file_path = f"{output_dir}/associations_{safe_name}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Association visualization saved to {file_path}")
    
    def visualize_network(self, selected_parts, top_n=3, output_dir='analysis_figures'):
        """
        Create a network visualization of part associations.
        
        Args:
            selected_parts (list): List of parts already selected
            top_n (int): Number of recommendations per selected part
            output_dir (str): Directory to save visualization
        """
        # Create graph
        G = nx.Graph()
        
        # Add selected parts as nodes
        for part in selected_parts:
            G.add_node(part, selected=True)
        
        # Add top recommendations for each selected part
        for selected_part in selected_parts:
            if selected_part in self.association_rules:
                top_recommendations = sorted(
                    [(part, rule['score']) for part, rule in 
                     self.association_rules[selected_part].items()
                     if part not in selected_parts],
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                
                for recommended_part, score in top_recommendations:
                    G.add_node(recommended_part, selected=False)
                    G.add_edge(selected_part, recommended_part, weight=score)
        
        # Create visualization
        plt.figure(figsize=(12, 12))
        
        # Node positions
        pos = nx.spring_layout(G, seed=42)
        
        # Node colors
        node_colors = ['red' if G.nodes[n].get('selected', False) else 'skyblue' 
                      for n in G.nodes()]
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Title
        plt.title('Part Association Network', fontsize=16)
        plt.axis('off')
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        parts_str = "_".join([re.sub(r'[\\/*?:"<>|]', '_', p)[:10] for p in selected_parts])
        file_path = f"{output_dir}/network_{parts_str}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Network visualization saved to {file_path}")

    def simulate_surveyor_session(self):
        """Simulate an interactive surveyor damage assessment session."""
        print("\n" + "="*80)
        print(" "*25 + "PART RECOMMENDATION SYSTEM DEMO")
        print("="*80)
        print("\nSimulating a surveyor damage assessment session with dynamic recommendations.")
        
        selected_parts = []
        
        while True:
            print("\nCurrently selected parts:")
            for i, part in enumerate(selected_parts, 1):
                print(f"  {i}. {part}")
            
            # Get recommendations based on current selections
            recommendations = self.get_recommendations(selected_parts)
            
            print("\nRecommended additional parts:")
            for i, (part, score) in enumerate(recommendations, 1):
                print(f"  {i}. {part} (score: {score:.2f})")
            
            # Create network visualization if parts are selected
            if selected_parts:
                self.visualize_network(selected_parts)
            
            # Get user input
            print("\nOptions:")
            print("  1. Add a recommended part")
            print("  2. Add a different part")
            print("  3. End assessment")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                part_choice = input(f"Enter part number to add (1-{len(recommendations)}): ")
                try:
                    idx = int(part_choice) - 1
                    if 0 <= idx < len(recommendations):
                        new_part = recommendations[idx][0]
                        selected_parts.append(new_part)
                        print(f"Added '{new_part}' to selected parts.")
                    else:
                        print("Invalid part number.")
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '2':
                # Show the top parts for manual selection
                top_parts = self.part_frequencies.most_common(20)
                print("\nTop 20 common parts:")
                for i, (part, count) in enumerate(top_parts, 1):
                    print(f"  {i}. {part} (frequency: {count})")
                
                part_choice = input(f"Enter part number to add (1-20), or enter custom part name: ")
                try:
                    idx = int(part_choice) - 1
                    if 0 <= idx < len(top_parts):
                        new_part = top_parts[idx][0]
                        selected_parts.append(new_part)
                        print(f"Added '{new_part}' to selected parts.")
                    else:
                        print("Invalid part number.")
                except ValueError:
                    # Assume user entered a custom part name
                    if part_choice in self.part_list:
                        selected_parts.append(part_choice)
                        print(f"Added '{part_choice}' to selected parts.")
                    else:
                        print(f"Part '{part_choice}' not found in database.")
            
            elif choice == '3':
                print("\nEnding damage assessment session.")
                break
            
            else:
                print("Invalid choice. Please try again.")
        
        print("\nFinal selected parts:")
        for i, part in enumerate(selected_parts, 1):
            print(f"  {i}. {part}")
        
        print("\nSimulation completed.")


def create_ui_mockup(output_dir='analysis_figures'):
    """Create a mockup of the Surveyor App UI with recommendation system"""
    # Create a figure for the UI mockup
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Set background color
    fig.patch.set_facecolor('#f0f0f0')
    ax.set_facecolor('#f0f0f0')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # App title
    ax.text(0.5, 0.97, 'AutoSure Insurance - Surveyor App', 
            fontsize=24, fontweight='bold', ha='center')
    
    # Date and time
    ax.text(0.95, 0.97, 'May 15, 2023 | 09:45 AM', 
            fontsize=10, ha='right')
    
    # Claim information
    ax.add_patch(plt.Rectangle((0.05, 0.85), 0.9, 0.08, fill=True, 
                              facecolor='white', edgecolor='#cccccc'))
    ax.text(0.07, 0.9, 'Claim #: 202112310018323', fontsize=12, fontweight='bold')
    ax.text(0.07, 0.87, 'Vehicle: Honda Civic (2020) | Policy: PL98765432', fontsize=10)
    ax.text(0.7, 0.89, 'Assessment Date: May 15, 2023', fontsize=10)
    
    # Damage Assessment Section
    ax.text(0.5, 0.82, 'Damage Assessment', fontsize=16, fontweight='bold', ha='center')
    
    # Selected parts section
    ax.add_patch(plt.Rectangle((0.05, 0.45), 0.4, 0.35, fill=True, 
                              facecolor='white', edgecolor='#cccccc'))
    ax.text(0.25, 0.77, 'Selected Damaged Parts', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Add selected parts
    selected_parts = [
        'Bumper Front Assembly',
        'Head Light Left'
    ]
    
    for i, part in enumerate(selected_parts):
        y_pos = 0.72 - i*0.05
        ax.add_patch(plt.Rectangle((0.08, y_pos-0.02), 0.34, 0.04, fill=True, 
                                  facecolor='#e6f2ff', edgecolor='#0066cc'))
        ax.text(0.1, y_pos, part, fontsize=10)
        ax.text(0.38, y_pos, '✕', fontsize=10, color='red', ha='center')
    
    # Add "+ Add part manually" button
    ax.add_patch(plt.Rectangle((0.08, 0.52), 0.34, 0.04, fill=True, 
                              facecolor='#f0f0f0', edgecolor='#cccccc', linestyle='--'))
    ax.text(0.25, 0.54, '+ Add part manually', fontsize=10, ha='center')
    
    # Recommendations section
    ax.add_patch(plt.Rectangle((0.55, 0.45), 0.4, 0.35, fill=True, 
                              facecolor='white', edgecolor='#cccccc'))
    ax.text(0.75, 0.77, 'Recommended Parts', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Add recommendation explanation
    ax.text(0.75, 0.73, 'Based on your selections, these parts are often damaged together:',
            fontsize=8, ha='center', style='italic')
    
    # Add recommended parts
    recommendations = [
        ('Fender Panel Front Left', 89.4),
        ('Bonnet|Hood Assembly', 71.2),
        ('Grille Radiator Upper', 68.9),
        ('Fender Panel Front Right', 62.5),
        ('Hinge Bonnet|Hood Left', 58.7)
    ]
    
    for i, (part, score) in enumerate(recommendations):
        y_pos = 0.68 - i*0.04
        
        # Background for recommendation
        confidence_width = score/100 * 0.34  # Width based on confidence
        ax.add_patch(plt.Rectangle((0.58, y_pos-0.015), confidence_width, 0.03, fill=True,
                                   facecolor='#d9f2d9', edgecolor=None, alpha=0.7))
        
        # Part name and confidence
        ax.text(0.6, y_pos, part, fontsize=9)
        ax.text(0.91, y_pos, f"{score:.1f}%", fontsize=8, ha='right')
        
        # Add button
        ax.add_patch(plt.Rectangle((0.88, y_pos-0.015), 0.04, 0.03, fill=True,
                                  facecolor='#007bff', edgecolor=None, alpha=0.8))
        ax.text(0.9, y_pos, '+', fontsize=9, ha='center', color='white')
    
    # Dynamic updating explanation
    ax.text(0.75, 0.48, 'Recommendations update dynamically as you select more parts',
            fontsize=8, ha='center', style='italic')
    
    # Additional information section
    ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.3, fill=True, 
                              facecolor='white', edgecolor='#cccccc'))
    ax.text(0.5, 0.37, 'Damage Assessment Details', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Add form fields
    fields = [
        ('Damage Severity:', 'Medium'),
        ('Estimated Repair Time:', '3-5 days'),
        ('Additional Notes:', 'Front collision with significant damage to bumper and left headlight.')
    ]
    
    for i, (label, value) in enumerate(fields):
        y_pos = 0.32 - i*0.06
        ax.text(0.1, y_pos, label, fontsize=10, fontweight='bold')
        
        # Add field
        ax.add_patch(plt.Rectangle((0.3, y_pos-0.02), 0.6, 0.04, fill=True, 
                                  facecolor='#f9f9f9', edgecolor='#cccccc'))
        ax.text(0.32, y_pos, value, fontsize=10)
    
    # Add submit button
    ax.add_patch(plt.Rectangle((0.7, 0.12), 0.2, 0.05, fill=True, 
                              facecolor='#28a745', edgecolor=None))
    ax.text(0.8, 0.145, 'Submit Assessment', fontsize=12, ha='center', color='white')
    
    # Save the mockup
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{output_dir}/surveyor_app_ui_mockup.png", dpi=300, bbox_inches='tight')
    print(f"UI mockup saved to {output_dir}/surveyor_app_ui_mockup.png")

# Example usage
if __name__ == "__main__":
    # Create UI mockup
    create_ui_mockup()
    
    # Initialize the recommendation system
    print("\nInitializing part recommendation system with top 500 most common parts...")
    recommender = PartRecommendationSystem('surveyor_data.csv', min_support=50, top_parts=500)
    
    # Visualize associations for some common parts
    common_parts = [part for part, _ in recommender.part_frequencies.most_common(3)]
    for part in common_parts:
        recommender.visualize_associations(part)
    
    # Demonstrate recommendations for specific scenarios
    print("\n" + "="*80)
    print(" "*25 + "RECOMMENDATION EXAMPLES")
    print("="*80)
    
    # Example 1: Front bumper damage
    selected_parts = ['Bumper Front Assembly']
    recommendations = recommender.get_recommendations(selected_parts)
    
    print(f"\nWhen 'Bumper Front Assembly' is selected, top 5 recommendations are:")
    for i, (part, score) in enumerate(recommendations, 1):
        print(f"  {i}. {part} (score: {score:.2f})")
    
    # Example 2: Front bumper and left headlight damage
    selected_parts = ['Bumper Front Assembly', 'Head Light Left']
    recommendations = recommender.get_recommendations(selected_parts)
    
    print(f"\nWhen 'Bumper Front Assembly' and 'Head Light Left' are selected, top 5 recommendations are:")
    for i, (part, score) in enumerate(recommendations, 1):
        print(f"  {i}. {part} (score: {score:.2f})")
    
    # Example 3: Rear collision
    selected_parts = ['Bumper Rear Assembly']
    recommendations = recommender.get_recommendations(selected_parts)
    
    print(f"\nWhen 'Bumper Rear Assembly' is selected, top 5 recommendations are:")
    for i, (part, score) in enumerate(recommendations, 1):
        print(f"  {i}. {part} (score: {score:.2f})")
    
    # Generate network visualizations for these examples
    recommender.visualize_network(['Bumper Front Assembly'])
    recommender.visualize_network(['Bumper Front Assembly', 'Head Light Left'])
    recommender.visualize_network(['Bumper Rear Assembly'])
    
    # Optionally run the interactive simulation
    simulate = input("\nWould you like to run the interactive simulation? (y/n): ")
    if simulate.lower() == 'y':
        recommender.simulate_surveyor_session() 
import numpy as np
from collections import defaultdict, Counter
import json
import matplotlib.pyplot as plt

class OffloadingStatsTracker:
    """
    Tracks offloading statistics for DNN nodes across episodes.
    Works alongside any RL agent that uses the action format:
    np.array([[level, decision]]) where decision: 0=edge, 1=cloudlet
    """
    
    def __init__(self, profiling_data):
        """
        Args:
            profiling_data: ProfilingData instance with layer/node info
        """
        self.profiling = profiling_data
        
        # Main statistics storage
        # Structure: level -> node_index -> {"edge": count, "cloud": count}
        self.offloading_counts = defaultdict(lambda: defaultdict(lambda: {"edge": 0, "cloud": 0, "total": 0}))
        
        # Per-episode tracking
        self.episode_stats = []
        
        # Action pattern frequencies
        self.action_pattern_counts = Counter()
        
        # Decision trends over time
        self.decision_trends = []
    
    def parse_action(self, action):
        """
        Parse an action array and return per-node decisions.
        
        Args:
            action: np.array([[level, decision], ...]) or similar
            
        Returns:
            List of tuples: [(level, node_index, decision), ...]
        """
        decisions = []
        
        if len(action.shape) == 2:  # Standard 2D array
            for i, row in enumerate(action):
                level = int(row[0])
                decision = int(row[1])
                # For multi-node levels, assume node index = position in array
                node_idx = i
                decisions.append((level, node_idx, decision))
        
        return decisions
    
    def track_action(self, action, episode_num=None, step_num=None):
        """
        Track a single action taken by the agent.
        
        Args:
            action: Action array from agent
            episode_num: Current episode number (optional)
            step_num: Current step number (optional)
        """
        # Parse the action
        decisions = self.parse_action(action)
        
        # Update per-node statistics
        for level, node_idx, decision in decisions:
            if decision == 0:  # Edge
                self.offloading_counts[level][node_idx]["edge"] += 1
            else:  # Cloudlet
                self.offloading_counts[level][node_idx]["cloud"] += 1
            
            self.offloading_counts[level][node_idx]["total"] += 1
        
        # Track action pattern (useful for frequency analysis)
        if decisions:
            # Create a pattern string for this action
            pattern = "_".join([f"{level}_{node}_{decision}" 
                              for level, node, decision in decisions])
            self.action_pattern_counts[pattern] += 1
        
        # Record trend data
        if episode_num is not None and step_num is not None:
            edge_count = sum(1 for _, _, d in decisions if d == 0)
            cloud_count = sum(1 for _, _, d in decisions if d == 1)
            self.decision_trends.append({
                "episode": episode_num,
                "step": step_num,
                "edge_decisions": edge_count,
                "cloud_decisions": cloud_count,
                "total_nodes": len(decisions)
            })
    
    def record_episode(self, episode_data):
        """
        Record complete episode statistics.
        
        Args:
            episode_data: Dict containing episode summary
        """
        self.episode_stats.append(episode_data)
    
    def get_node_offloading_rate(self, level, node_idx):
        """
        Get offloading rate for a specific node.
        
        Returns: Percentage of times this node was offloaded to cloudlet
        """
        if level not in self.offloading_counts or node_idx not in self.offloading_counts[level]:
            return 0.0
        
        stats = self.offloading_counts[level][node_idx]
        if stats["total"] == 0:
            return 0.0
        
        return (stats["cloud"] / stats["total"]) * 100
    
    def get_level_summary(self, level):
        """
        Get summary statistics for a specific level.
        
        Returns: Dict with level statistics
        """
        if level not in self.offloading_counts:
            return {"edge": 0, "cloud": 0, "total": 0, "offloading_rate": 0.0}
        
        edge_total = 0
        cloud_total = 0
        
        for node_stats in self.offloading_counts[level].values():
            edge_total += node_stats["edge"]
            cloud_total += node_stats["cloud"]
        
        total = edge_total + cloud_total
        offloading_rate = (cloud_total / total * 100) if total > 0 else 0.0
        
        return {
            "edge": edge_total,
            "cloud": cloud_total,
            "total": total,
            "offloading_rate": offloading_rate,
            "node_count": len(self.offloading_counts[level])
        }
    
    def print_summary(self):
        """Print comprehensive offloading statistics."""
        print("\n" + "="*70)
        print("OFFLOADING STATISTICS SUMMARY")
        print("="*70)
        
        # Overall statistics
        total_edge = 0
        total_cloud = 0
        
        print("\nPer-Level Offloading Rates:")
        print("-"*40)
        for level in sorted(self.offloading_counts.keys()):
            level_summary = self.get_level_summary(level)
            total_edge += level_summary["edge"]
            total_cloud += level_summary["cloud"]
            
            print(f"Level {level:2d}: "
                  f"Edge={level_summary['edge']:6d} | "
                  f"Cloud={level_summary['cloud']:6d} | "
                  f"Rate={level_summary['offloading_rate']:6.2f}% | "
                  f"Nodes={level_summary['node_count']}")
        
        # Overall statistics
        total_decisions = total_edge + total_cloud
        overall_rate = (total_cloud / total_decisions * 100) if total_decisions > 0 else 0.0
        
        print("\n" + "-"*70)
        print(f"OVERALL: Edge={total_edge:,} | Cloud={total_cloud:,} | "
              f"Total={total_decisions:,} | Offloading Rate={overall_rate:.2f}%")
        print("="*70)
    
    def get_heatmap_data(self):
        """
        Prepare data for offloading heatmap visualization.
        
        Returns:
            Array with offloading rates for each node
        """
        max_level = max(self.offloading_counts.keys()) if self.offloading_counts else 0
        max_nodes = max(len(nodes) for nodes in self.offloading_counts.values()) if self.offloading_counts else 0
        
        heatmap = np.zeros((max_level + 1, max_nodes))
        
        for level in self.offloading_counts:
            for node_idx, stats in self.offloading_counts[level].items():
                if stats["total"] > 0:
                    rate = stats["cloud"] / stats["total"]
                    heatmap[level, node_idx] = rate
        
        return heatmap
    
    def plot_offloading_heatmap(self, save_path=None):
        """Create a heatmap visualization of offloading decisions."""
        heatmap_data = self.get_heatmap_data()
        
        if heatmap_data.size == 0:
            print("No data to plot")
            return
        
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Offloading Rate (0=Edge, 1=Cloudlet)')
        plt.xlabel('Node Index')
        plt.ylabel('DNN Level')
        plt.title('Node Offloading Heatmap')
        
        # Add text annotations
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                if heatmap_data[i, j] > 0:
                    plt.text(j, i, f'{heatmap_data[i, j]:.2f}', 
                            ha='center', va='center', color='black', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_offloading_trends(self, save_path=None):
        """Plot offloading decision trends over episodes."""
        if not self.decision_trends:
            print("No trend data available")
            return
        
        # Group by episode
        episodes = sorted(set(d['episode'] for d in self.decision_trends))
        edge_by_episode = []
        cloud_by_episode = []
        
        for ep in episodes:
            ep_data = [d for d in self.decision_trends if d['episode'] == ep]
            edge_total = sum(d['edge_decisions'] for d in ep_data)
            cloud_total = sum(d['cloud_decisions'] for d in ep_data)
            edge_by_episode.append(edge_total)
            cloud_by_episode.append(cloud_total)
        
        plt.figure(figsize=(12, 6))
        x = range(len(episodes))
        
        plt.plot(x, edge_by_episode, 'b-', label='Edge Decisions', linewidth=2)
        plt.plot(x, cloud_by_episode, 'r-', label='Cloud Decisions', linewidth=2)
        plt.plot(x, np.array(edge_by_episode) + np.array(cloud_by_episode), 
                'g--', label='Total Decisions', linewidth=1.5, alpha=0.7)
        
        plt.xlabel('Episode')
        plt.ylabel('Number of Decisions')
        plt.title('Offloading Decision Trends Over Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trend plot saved to {save_path}")
        
        plt.show()
    
    def save_statistics(self, filename="offloading_stats.json"):
        """Save statistics to JSON file."""
        stats_dict = {
            "per_node_stats": {},
            "per_level_summary": {},
            "action_patterns": dict(self.action_pattern_counts.most_common(20)),
            "total_episodes": len(self.episode_stats)
        }
        
        # Save per-node statistics
        for level in self.offloading_counts:
            stats_dict["per_node_stats"][str(level)] = {
                str(node): stats 
                for node, stats in self.offloading_counts[level].items()
            }
        
        # Save per-level summary
        for level in sorted(self.offloading_counts.keys()):
            stats_dict["per_level_summary"][str(level)] = self.get_level_summary(level)
        
        with open(filename, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"Statistics saved to {filename}")
    
    def load_statistics(self, filename="offloading_stats.json"):
        """Load statistics from JSON file."""
        try:
            with open(filename, 'r') as f:
                stats_dict = json.load(f)
            
            # Load per-node statistics
            for level_str, node_dict in stats_dict.get("per_node_stats", {}).items():
                level = int(level_str)
                for node_str, stats in node_dict.items():
                    node_idx = int(node_str)
                    self.offloading_counts[level][node_idx] = stats
            
            print(f"Statistics loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Statistics file {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading statistics: {e}")
            return False
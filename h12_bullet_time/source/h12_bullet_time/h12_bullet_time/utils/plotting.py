import matplotlib.pyplot as plt
import numpy as np
import argparse
import json

COLORS = {"TOF": "#E63946", "CAP": "#457B9D", "CAP_TOF": "#2A9D8F"}
BIN_COLORS = ["#E09489", "#E0DB78", "#BDE0C5", "#9BE1AF", "#78E09C", "#5BE186", "#37EF78"]
CONTACT_COLOR = "#8B0000"  # Dark red for terminated/contact episodes

BIN_THRESHOLDS = [0.0001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
BIN_LABELS = ["< 0.0001m", "< 0.01m", "< 0.1m", "< 0.2m", "< 0.3m", "< 0.4m", "< 0.5m"]

def extract_sensor_data(data, sensor_type):
    """Extract and sort data for a specific sensor type."""
    sensor_data = [
        exp for exp in data
        if exp["params"]["ABLATION_SENSOR_TYPE"] == sensor_type and exp["test_metrics"]
    ]
    sensor_data.sort(key=lambda x: x["params"]["ABLATION_MAX_RANGE"])
    return sensor_data


def plot_distance_bins_area(data, sensor_type, ax):
    """Plot overlapping area chart for distance bins."""
    sensor_data = extract_sensor_data(data, sensor_type)
    if not sensor_data:
        ax.set_title(f"{sensor_type} Sensor - No Data")
        return
    
    ranges = [exp["params"]["ABLATION_MAX_RANGE"] for exp in sensor_data]
    
    # Plot "Contact" bin first (episodes that didn't stay alive) - appears as top/background layer
    contact_counts = [
        exp["test_metrics"]["total_episodes"] - exp["test_metrics"].get("stayed_alive_count", 0)
        for exp in sensor_data
    ]
    contact_rate = [contact_counts[i] / sensor_data[i]["test_metrics"]["total_episodes"] for i in range(len(sensor_data))]
    
    # Plot distance threshold bins (layered on top of contact)
    for i, threshold in enumerate(reversed(BIN_THRESHOLDS)):
        counts = [exp["test_metrics"].get(f"dist_min_below_{threshold}", 0) for exp in sensor_data]
        rate = [counts[i] / sensor_data[i]["test_metrics"]["total_episodes"] for i in range(len(sensor_data))]
        color_idx = len(BIN_THRESHOLDS) - 1 - i
        ax.fill_between(ranges, rate, alpha=0.7, label=BIN_LABELS[color_idx], color=BIN_COLORS[color_idx])
        ax.plot(ranges, rate, color=BIN_COLORS[color_idx], linewidth=1.5)

    ax.fill_between(ranges, 0, contact_rate, alpha=1.0, label="Contact (terminated)", color=CONTACT_COLOR)
    ax.plot(ranges, contact_rate, color=CONTACT_COLOR, linewidth=1.5, zorder=10)
    
    ax.set(
        xlabel="Sensing Max Range (m)",
        ylabel=f"Rate (% of {sensor_data[0]['test_metrics']['total_episodes']} episodes)",
        title=f"{sensor_type} Sensor - Closest Approach",
        xlim=(min(ranges), max(ranges)),
    )
    ax.grid(True, alpha=0.3)


def plot_comparison(data, ax, metric_key, ylabel, title, show_std=False, std_key=None):
    """Generic comparison plot for any metric."""
    # Collect all unique range values from all sensor types for x-ticks
    all_ranges = set()
    
    for sensor_type, color in COLORS.items():
        sensor_data = extract_sensor_data(data, sensor_type)
        if not sensor_data:
            continue
        
        ranges = np.array([exp["params"]["ABLATION_MAX_RANGE"] for exp in sensor_data])
        all_ranges.update(ranges)
        
        # Handle top-level vs nested metrics
        values = np.array([
            exp.get(metric_key) if metric_key in exp else exp["test_metrics"].get(metric_key, 0)
            for exp in sensor_data
        ])
        
        ax.plot(ranges, values, 'o-', color=color, label=sensor_type, linewidth=2, markersize=6)
        
        # Try to find std key - either use provided std_key, or infer from metric_key
        std_key_to_use = std_key
        if not std_key_to_use and metric_key.startswith("mean_"):
            # Infer std key from mean key (e.g., "mean_reward" -> "std_reward")
            std_key_to_use = metric_key.replace("mean_", "std_")
        
        # Plot std area if available (either explicitly requested or inferred)
        if std_key_to_use and sensor_data:
            # Check if std_key exists in the data
            if std_key_to_use in sensor_data[0].get("test_metrics", {}):
                std = np.array([exp["test_metrics"].get(std_key_to_use, 0) for exp in sensor_data])
                ax.fill_between(ranges, values - std, values + std, color=color, alpha=0.2)
    
    # Set x-axis to log scale and use exact logged values as ticks
    ax.set(xlabel="Sensing Max Range in meters (log scale)", ylabel=ylabel, title=title)
    ax.set_xscale('log')
    if all_ranges:
        ax.set_xticks(sorted(all_ranges))
        ax.set_xticklabels([f"{r:.2f}" if r < 1 else f"{r:.1f}" for r in sorted(all_ranges)], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.5, which='both')

LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
MARKERS = ['o', 's', 'D', 'P', 'X', 'v', '^', '<', '>', '1', '2', '3', '4', '8']

def plot_param_sweep(data, ax, metric_key, vary_param, fixed_param, fixed_value, ylabel, title,
                     vary_label=None, fixed_label=None):
    """Plot a metric vs sensor range, varying one param while holding another fixed.
    
    Args:
        data: Ablation experiment data.
        ax: Matplotlib axis to plot on.
        metric_key: The metric to plot on y-axis (e.g., "mean_reward").
        vary_param: The parameter to create separate lines for (e.g., "ABLATION_PROXIMITY_SCALE").
        fixed_param: The parameter to hold constant (e.g., "ABLATION_PROJECTILE_MASS").
        fixed_value: The value to fix the fixed_param at.
        ylabel: Y-axis label.
        title: Plot title.
        vary_label: Optional human-readable name for the varying parameter.
        fixed_label: Optional human-readable name for the fixed parameter.
    """
    if vary_label is None:
        vary_label = vary_param.replace("ABLATION_", "").replace("_", " ").title()
    if fixed_label is None:
        fixed_label = fixed_param.replace("ABLATION_", "").replace("_", " ").title()
    
    for sensor_type, color in COLORS.items():
        sensor_data = extract_sensor_data(data, sensor_type)
        if not sensor_data:
            continue
        
        # Filter by fixed parameter value
        sensor_data = [exp for exp in sensor_data if exp["params"].get(fixed_param) == fixed_value]
        if not sensor_data:
            continue
        
        # Get unique values of the varying parameter
        vary_values = sorted(set(exp["params"].get(vary_param) for exp in sensor_data))
        
        for i, vary_val in enumerate(vary_values):
            linestyle = LINE_STYLES[i % len(LINE_STYLES)]
            marker = MARKERS[i % len(MARKERS)]
            
            # Filter data for this vary value
            filtered = [exp for exp in sensor_data if exp["params"].get(vary_param) == vary_val]
            if not filtered:
                continue
            
            # Sort by range and extract values
            filtered.sort(key=lambda x: x["params"]["ABLATION_MAX_RANGE"])
            ranges = np.array([exp["params"]["ABLATION_MAX_RANGE"] for exp in filtered])
            
            # Handle top-level vs nested metrics
            values = np.array([
                exp.get(metric_key) if metric_key in exp else exp["test_metrics"].get(metric_key, 0)
                for exp in filtered
            ])
            
            label = f"{sensor_type} ({vary_label}={vary_val})"
            ax.plot(ranges, values, marker=marker, linestyle=linestyle, color=color, 
                    label=label, linewidth=2, markersize=5)
    
    ax.set(xlabel="Sensing Max Range (m)", ylabel=ylabel, title=f"{title}\n({fixed_label}={fixed_value})")
    ax.grid(True, alpha=0.3)


def get_unique_param_values(data, param_name):
    """Get sorted unique values for a parameter across all experiments."""
    return sorted(set(exp["params"].get(param_name) for exp in data if param_name in exp["params"]))


def create_all_plots(data, save_path=None):
    """Create all ablation result plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    # Set font to Times New Roman for all plots
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    
    # # Figure 1: Distance bin area charts
    figures = []
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 5))
    fig1.suptitle("Distance Distribution by Sensor Type", fontsize=15, fontweight='bold', y=1.02)
    for ax, sensor in zip(axes1, ["TOF", "CAP", "CAP_TOF"]):
        plot_distance_bins_area(data, sensor, ax)
    # Shared legend for both area charts at the bottom center
    handles, labels = axes1[0].get_legend_handles_labels()
    if handles:  # Only add legend if there's data
        fig1.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=max(1, len(labels)),
            fontsize=9,
            frameon=False,
        )
    fig1.tight_layout(rect=[0, 0.05, 1, 1])
    figures.append((fig1, "distance_bins_area"))
    # Figure 2: Comparison plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 5))
    fig2.suptitle("Sensor Performance Comparison", fontsize=15, fontweight='bold', y=1.02)
    
    comparisons = [
        ("success", "Success Rate", "a) Success Rate vs Sensing Range", False, None),
        ("median_closest_distance", "Median Closest Distance", "b) Median Closest Distance vs Sensing Range", False, None),
        ("mean_episode_length", "Mean Episode Length", "c) Mean Episode Length vs Sensing Range", False, None),
        ("mean_reward", "Mean Reward", "d) Mean Reward vs Sensing Range", True, "std_reward"),
    ]
    for ax, (metric, ylabel, title, show_std, std_key) in zip(axes2.flat, comparisons):
        plot_comparison(data, ax, metric, ylabel, title, show_std, std_key)
    
    axes2[0, 0].set_ylim(0, 1.05)  # Success rate specific
    
    # Shared legend for all comparison plots at the bottom center
    handles, labels = axes2[0, 0].get_legend_handles_labels()
    if handles:  # Only add legend if there's data
        # Remove individual legends from all subplots
        for ax in axes2.flat:
            legend = ax.get_legend()
            if legend:
                legend.remove()
        # Add shared legend at bottom
        fig2.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=max(1, len(labels)),
            fontsize=10,
            frameon=False,
        )
    fig2.tight_layout(rect=[0, 0.05, 1, 1])
    figures.append((fig2, "comparison"))
    # Figure 3: Parameter sweep plots
    # For each pair of params, fix one and vary the other
    # params = [
    #     ("ABLATION_PROXIMITY_SCALE", "Prox Scale"),
    #     ("ABLATION_PROJECTILE_MASS", "Proj Mass"),
    # ]
    
    # figures = []
    # for fixed_param, fixed_label in params:
    #     # Get all other params to vary
    #     other_params = [(p, l) for p, l in params if p != fixed_param]
        
    #     for vary_param, vary_label in other_params:
    #         # Get unique values of the fixed parameter
    #         fixed_values = get_unique_param_values(data, fixed_param)
            
    #         # Create a figure with subplots: rows = fixed values, cols = metrics
    #         n_fixed = len(fixed_values)
    #         n_metrics = len(comparisons)
            
    #         fig, axes = plt.subplots(n_fixed, n_metrics, figsize=(5 * n_metrics, 4 * n_fixed))
    #         fig.suptitle(f"Mean Reward by {vary_label} (varying {fixed_label})", 
    #                     fontsize=14, fontweight='bold', y=1.01)
            
    #         # Handle single row case
    #         if n_fixed == 1:
    #             axes = axes.reshape(1, -1)
            
    #         for row, fixed_val in enumerate(fixed_values):
    #             for col, (metric, ylabel, title, show_std, std_key) in enumerate(comparisons):
    #                 ax = axes[row, col]
    #                 plot_param_sweep(data, ax, metric, vary_param, fixed_param, fixed_val,
    #                                ylabel, title, vary_label=vary_label, fixed_label=fixed_label)
            
    #         # Add single shared legend for the figure
    #         handles, labels = axes[0, 0].get_legend_handles_labels()
    #         fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
    #                   ncol=len(labels), fontsize=9, frameon=True)
            
    #         fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    #         figures.append((fig, f"{fixed_param}_fixed_vary_{vary_param}"))
    
    if save_path:
        for fig, name in figures:
            filename = f"{save_path}_{name}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--data", type=str, required=True, help="Path to ablation results JSON")
    parser.add_argument("--save", type=str, default=None, help="Base path to save plots (optional)")
    args = parser.parse_args()
    
    save_path = args.save if args.save else args.data.rsplit('.', 1)[0]
    
    with open(args.data) as f:
        data = json.load(f)
    
    create_all_plots(data, save_path=save_path)

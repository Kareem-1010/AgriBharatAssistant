import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

def generate_benchmark_plots():
    results_path = Path(__file__).resolve().parent / "evaluation_results.json"
    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"Cannot find results file: {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Category Performance Bar Plot
    categories = data.get("category_breakdown", {})
    if categories:
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        names = [k.replace("_", " ").title() for k, _ in sorted_cats]
        scores = [v * 100 for _, v in sorted_cats]
        
        bars = ax.barh(names, scores, color="#2e7d32", height=0.6)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Benchmark Composite Score (%)", fontweight="bold")
        ax.set_title("KrishiMitra AI: Evaluation Performance by Agricultural Domain (45 Test Cases)", fontsize=13, fontweight="bold", pad=12)
        
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold", color="#1b5e20")
        
        plt.tight_layout()
        cat_plot_path = plots_dir / "benchmark_categories.png"
        plt.savefig(cat_plot_path, dpi=300)
        plt.close()
        print(f"Saved: {cat_plot_path}")

    # 2. Multilingual Performance Plot
    languages = data.get("language_breakdown", {})
    if languages:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        lang_names = {
            "hi": "Hindi (हिन्दी)",
            "en": "English",
            "te": "Telugu (తెలుగు)",
            "ta": "Tamil (தமிழ்)",
            "mr": "Marathi (मराठी)",
            "bn": "Bengali (বাংলা)"
        }
        sorted_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)
        labels = [lang_names.get(k, k) for k, _ in sorted_langs]
        scores = [v * 100 for _, v in sorted_langs]

        bars = ax.bar(labels, scores, color="#0288d1", width=0.55)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Composite Score (%)", fontweight="bold")
        ax.set_title("Multilingual Benchmark Performance across Indic Languages", fontsize=13, fontweight="bold", pad=12)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#01579b")

        plt.tight_layout()
        lang_plot_path = plots_dir / "language_performance.png"
        plt.savefig(lang_plot_path, dpi=300)
        plt.close()
        print(f"Saved: {lang_plot_path}")

    # 3. Aggregate RAG Triad Radar / Metrics Plot
    agg = data.get("aggregate_metrics", {})
    if agg:
        fig, ax = plt.subplots(figsize=(8, 4))
        metric_labels = [
            "Context Recall@K",
            "Groundedness\n(Faithfulness)",
            "Keyword &\nFactual Precision",
            "Overall Benchmark\nComposite"
        ]
        values = [
            agg.get("mean_context_recall", 0) * 100,
            agg.get("mean_groundedness", 0) * 100,
            agg.get("mean_keyword_precision", 0) * 100,
            agg.get("mean_overall_score", 0) * 100
        ]
        colors = ["#388e3c", "#f57c00", "#7b1fa2", "#1976d2"]
        bars = ax.bar(metric_labels, values, color=colors, width=0.5)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Percentage (%)", fontweight="bold")
        ax.set_title("KrishiMitra AI: Core GenAI & RAG Triad Metric Scores", fontsize=13, fontweight="bold", pad=12)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        metrics_plot_path = plots_dir / "rag_triad_metrics.png"
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
        print(f"Saved: {metrics_plot_path}")

if __name__ == "__main__":
    generate_benchmark_plots()

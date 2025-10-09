import csv
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def load_csv(csv_path):
    """Load CSV into a list of dicts"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def compute_category_distribution(data):
    object_category_counts = Counter()
    surface_category_counts = Counter()
    per_camera_object_counts = defaultdict(Counter)
    per_camera_surface_counts = defaultdict(Counter)

    for row in data:
        camera = row["camera"]

        objects = json.loads(row["object_categories"])
        surfaces = json.loads(row["surface_categories"])

        object_category_counts.update(objects)
        surface_category_counts.update(surfaces)

        per_camera_object_counts[camera].update(objects)
        per_camera_surface_counts[camera].update(surfaces)

    return {
        "object_category_counts": dict(object_category_counts),
        "surface_category_counts": dict(surface_category_counts),
        "per_camera_object_counts": {k: dict(v) for k, v in per_camera_object_counts.items()},
        "per_camera_surface_counts": {k: dict(v) for k, v in per_camera_surface_counts.items()}
    }


def plot_category_distribution(dist_dict, title, ax=None):
    categories = list(dist_dict.keys())
    counts = [dist_dict[c] for c in categories]

    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 8))
    
    ax.bar(range(len(categories)), counts, color='skyblue')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c for c in categories], rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    
    plt.tight_layout()


def generate_data_quality_report(csv_paths, output_pdf="data_quality_report.pdf"):
    """
    csv_paths: dict of {split_name: csv_path}, e.g., {"train": "train.csv", "val": "val.csv"}
    """
    with PdfPages(output_pdf) as pdf:
        for split_name, csv_path in csv_paths.items():
            data = load_csv(csv_path)
            dist = compute_category_distribution(data)

            fig, ax = plt.subplots(figsize=(12, 6))
            plot_category_distribution(dist["object_category_counts"], f"{split_name} - Overall Object Categories", ax=ax)
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 6))
            plot_category_distribution(dist["surface_category_counts"], f"{split_name} - Overall Surface Categories", ax=ax)
            pdf.savefig(fig)
            plt.close(fig)

            for cam, counter in dist["per_camera_object_counts"].items():
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_category_distribution(counter, f"{split_name} - Object Categories in Camera {cam}", ax=ax)
                pdf.savefig(fig)
                plt.close(fig)

            for cam, counter in dist["per_camera_surface_counts"].items():
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_category_distribution(counter, f"{split_name} - Surface Categories in Camera {cam}", ax=ax)
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Data quality report saved to {output_pdf}")


if __name__ == "__main__":
    csv_paths = {
        "train": "nuimages_csv_splits/train.csv",
        "val": "nuimages_csv_splits/val.csv",
        "test": "nuimages_csv_splits/test.csv"
    }
    generate_data_quality_report(csv_paths)

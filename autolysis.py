# ///

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Ensure AIPROXY_TOKEN is set
api_key = os.getenv("AIPROXY_TOKEN")
if not api_key:
    raise EnvironmentError("AIPROXY_TOKEN environment variable not set.")
openai.api_key = api_key

def analyze_csv(file_path, output_dir="goodreads"):
    """Main function to analyze the CSV file."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load the dataset
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Step 2: Basic summary
    print("Performing basic analysis...")
    info = df.info()
    summary = df.describe().to_string()

    # Step 3: Generate plots
    sns.set(style="whitegrid")

    # Plot 1: Distribution of Numeric Data
    plt.figure(figsize=(10, 6))
    sns.histplot(df.select_dtypes(include="number"), kde=True)
    plt.title("Distribution of Numeric Data")
    plt.savefig(os.path.join(output_dir, "distribution.png"))
    print(f"Saved plot as '{os.path.join(output_dir, 'distribution.png')}'.")

    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(output_dir, "correlation.png"))
    print(f"Saved plot as '{os.path.join(output_dir, 'correlation.png')}'.")

    # Plot 3: Pairplot of Numeric Data
    plt.figure(figsize=(10, 6))
    sns.pairplot(df.select_dtypes(include="number"))
    plt.title("Pairplot of Numeric Data")
    plt.savefig(os.path.join(output_dir, "pairplot.png"))
    print(f"Saved plot as '{os.path.join(output_dir, 'pairplot.png')}'.")

    # Step 4: Narrate insights (LLM)
    prompt = f"Summarize the dataset analysis for: {file_path}. Here is the basic summary:\n{summary}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    narrative = response['choices'][0]['message']['content']
    print("Narrative:", narrative)

    # Step 5: Generate README.md
    readme_content = f"# Dataset Analysis\n\n## Summary\n\n{info}\n\n{summary}\n\n## Narrative\n\n{narrative}\n\n"
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as readme_file:
        readme_file.write(readme_content)
    print(f"Saved README as '{readme_path}'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
    else:
        analyze_csv(sys.argv[1])

# ///
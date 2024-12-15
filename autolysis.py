import os
import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import openai

# OpenAI API configuration (assuming AIPROXY_TOKEN environment variable is set)
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
api_key = os.getenv("AIPROXY_TOKEN")
if not api_key:
    raise EnvironmentError("AIPROXY_TOKEN environment variable not set.")
openai.api_key = api_key

def load_data(file_path: str) -> str:
    """
    Loads CSV data with automatic encoding detection.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        str: Detected encoding used to read the file.
    """

    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'windows-1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {file_path} with any tried encodings.")

def create_directory(output_dir: Path) -> None:
    """
    Creates the output directory if it doesn't exist.

    Args:
        output_dir (Path): Path to the output directory.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

def analyze_dataset(file_path: str) -> (pd.DataFrame, dict):
    """
    Performs basic data analysis on the CSV dataset.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing the pandas DataFrame and a dictionary summarizing the data.
    """

    encoding = load_data(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    summary = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "summary_stats": df.describe(include="all").to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }
    return df, summary

def generate_visualizations(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """
    Generates visualizations (correlation heatmap) and saves them as PNGs.

    Args:
        df (pd.DataFrame): The pandas DataFrame.
        output_dir (Path): Path to the output directory.

    Returns:
        list[Path]: A list of paths to the generated visualizations.
    """

    visualizations = []

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.title("Correlation Heatmap")
    plt.savefig(heatmap_path)
    plt.close()
    visualizations.append(heatmap_path)

    return visualizations

def generate_readme(summary: dict, image_paths: list[Path], output_dir: Path) -> None:
    """
    Generates a Markdown README file with the data analysis summary and visualizations.

    Args:
        summary (dict): Dictionary containing the data summary.
        image_paths (list[Path]): List of paths to the generated visualizations.
        output_dir (Path): Path to the output directory.
    """

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("# Automated Analysis\n\n")
        f.write("## Summary\n")
        f.write(f"### Columns:\n{', '.join(summary['columns'])}\n\n")
        f.write("### Missing Values:\n")
        for col, missing in summary["missing_values"].items():
            f.write(f"- {col}: {missing}\n")
        f.write("\n### Visualizations\n")
        for image_path in image_paths:
            f.write(f"![Visualization]({image_path.name})\n\n")

def query_llm(summary: dict) -> str:
    """
    Generates narrative insights using the OpenAI API.

    Args:
        summary (dict): Dictionary containing the data summary.

    Returns:
        str: Narrative insights generated by the LLM.
    """

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    prompt = f"Provide a detailed analysis based on the following data summary: {summary}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return "Narrative generation failed due to an error."

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    dataset_name = Path(csv_file).stem
    output_dir = Path.cwd() / dataset_name

    # Ensure output directory exists
    create_directory(output_dir)

    # Load Dataset
    encoding = load_data(csv_file)

    # Analyze dataset
    df, summary = analyze_dataset(csv_file)

    # Generate visualizations
    image_paths = generate_visualizations(df, output_dir)

    # Query LLM for narrative insights
    insights = query_llm(summary)

    # Add insights to README
    generate_readme(summary, image_paths, output_dir)

    print(f"Analysis complete. Output saved to: {output_dir}")

if __name__ == "__main__":
    main()

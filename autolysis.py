# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "chardet>=5.2.0",
#   "matplotlib>=3.9.3",
#   "numpy>=2.2.0",
#   "openai>=1.57.2",
#   "pandas>=2.2.3",
#   "python-dotenv>=1.0.1",
#   "requests>=2.32.3",
#   "scikit-learn>=1.6.0",
#   "seaborn>=0.13.2",
# ]
# ///

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import chardet
import base64

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable is not set.")

BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def query_chat_completion(prompt, model="gpt-4o-mini"):
    """Send a chat prompt to the LLM and return the response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response content returned.")
    except requests.RequestException as e:
        raise Exception(f"Error during LLM query: {e}")


def detect_file_encoding(filepath):
    """Detect the encoding of a file."""
    with open(filepath, "rb") as file:
        result = chardet.detect(file.read(100000))  # Read the first 100,000 bytes
        return result["encoding"]

def load_data(filename):
    """Load CSV data into a Pandas DataFrame, handling file encoding with fallbacks."""
    try:
        # Detect encoding
        encoding = detect_file_encoding(filename)
        print(f"Detected encoding for {filename}: {encoding}")

        # Try reading the file with the detected encoding
        return pd.read_csv(filename, encoding=encoding)
    except Exception as primary_error:
        print(f"Primary encoding {encoding} failed: {primary_error}")

        # Fallback encodings
        fallback_encodings = ["utf-8-sig", "latin1"]
        for fallback in fallback_encodings:
            try:
                print(f"Trying fallback encoding: {fallback}")
                return pd.read_csv(filename, encoding=fallback)
            except Exception as fallback_error:
                print(f"Fallback encoding {fallback} failed: {fallback_error}")

        # Raise error if all attempts fail
        raise ValueError(f"Failed to load file {filename} with any encoding.")

def generic_analysis(df):
    """Perform generic analysis on the dataset."""
    analysis = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_stats": df.describe(include="all").to_dict(),
        "variance": df.var(numeric_only=True).to_dict(),
        "skewness": df.skew(numeric_only=True).to_dict()
    }
    return analysis

def preprocess_data(df):
    """Preprocess data to handle missing values."""
    numeric_df = df.select_dtypes(include=['float', 'int'])
    imputer = SimpleImputer(strategy='mean')  # Replace missing values with the mean
    numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
    return numeric_df_imputed

def preprocess_for_visualization(df, max_rows=1000):
    """Limit the dataset to a subset for faster visualizations."""
    if df.shape[0] > max_rows:
        return df.sample(max_rows, random_state=42)
    return df

def detect_feature_types(df):
    """Detect feature types for special analyses."""
    return {
        "time_series": df.select_dtypes(include=['datetime']).columns.tolist(),
        "geographic": [col for col in df.columns if any(geo in col.lower() for geo in ["latitude", "longitude", "region", "country"])],
        "network": [col for col in df.columns if "source" in col.lower() or "target" in col.lower()],
        "cluster": df.select_dtypes(include=['float', 'int']).columns.tolist()  # Numeric features for clustering
    }

def perform_special_analyses(df, feature_types):
    """Perform special analyses based on feature types."""
    analyses = {}

    # Time Series Analysis
    if feature_types["time_series"]:
        analyses["time_series"] = [
            f"Time-series features detected: {', '.join(feature_types['time_series'])}. "
            "These can be used to observe trends or forecast future patterns."
        ]
    else:
        analyses["time_series"] = ["No time-series features detected."]

    # Geographic Analysis
    if len(feature_types["geographic"]) >= 2:
        analyses["geographic"] = [
            f"Geographic features detected: {', '.join(feature_types['geographic'][:2])}. "
            "These can be used to visualize or analyze spatial distributions."
        ]
    else:
        analyses["geographic"] = ["No geographic features detected."]

    # Network Analysis
    if len(feature_types["network"]) >= 2:
        analyses["network"] = [
            f"Network relationships detected between {feature_types['network'][0]} and {feature_types['network'][1]}. "
            "These can be analyzed for connectivity or collaborations."
        ]
    else:
        analyses["network"] = ["No network features detected."]

    # Cluster Analysis
    if len(feature_types["cluster"]) > 1:
        analyses["cluster"] = [
            "Cluster analysis is feasible with the available numeric features. "
            "This could help identify natural groupings in the data."
        ]
    else:
        analyses["cluster"] = ["Not enough numeric features for cluster analysis."]

    return analyses

def create_visualizations(df):
    """Generate and save visualizations based on data."""
    numeric_df = preprocess_data(df)
    visualization_df = preprocess_for_visualization(numeric_df)

    # Correlation Heatmap
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", cbar_kws={'shrink': 0.8})
        plt.title("Correlation Heatmap", fontsize=16)
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.savefig("correlation_heatmap.png")
        plt.close()

    # Outlier Detection with Seaborn Scatter Plot
    if visualization_df.shape[1] > 1:
        model = IsolationForest(random_state=42)
        visualization_df['outlier_score'] = model.fit_predict(visualization_df)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=visualization_df, x=visualization_df.columns[0], y=visualization_df.columns[1], hue='outlier_score', palette="Set1")
        plt.title("Outlier Detection (Scatter Plot)", fontsize=16)
        plt.xlabel(visualization_df.columns[0])
        plt.ylabel(visualization_df.columns[1])
        plt.legend(title="Outliers")
        plt.savefig("outlier_detection.png")
        plt.close()

    # Pair Plot for Relationship Analysis (limited columns)
    if visualization_df.shape[1] > 1:
        selected_columns = visualization_df.columns[:5]  # Limit to first 5 numeric columns
        sns.pairplot(visualization_df[selected_columns], palette="husl")
        plt.savefig("pairplot_analysis.png")
        plt.close()

    # Distribution Plot
    for col in visualization_df.columns[:5]:  # Limit to first 5 numeric columns
        plt.figure(figsize=(8, 6))
        sns.histplot(visualization_df[col], kde=True, color="skyblue")
        plt.title(f"Distribution of {col}", fontsize=16)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(f"distribution_{col}.png")
        plt.close()

    return ["correlation_heatmap.png", "outlier_detection.png", "pairplot_analysis.png"]

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def analyze_image_with_vision_api(image_path, model="gpt-4o-mini"):
    """Analyze an image using the OpenAI Vision API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Analyze this image."}],
    "image": image_to_base64(image_path),
    }
    response = requests.post(BASE_URL, headers=headers, json=payload)

def narrate_story(summary, insights, charts, special_analyses):
    """Generate a narrative story about the analysis, including image analysis."""
    image_analyses = []
    for chart in charts:
        analysis = analyze_image_with_vision_api(chart)
        image_analyses.append(f"Analysis for {chart}: {analysis}")

    special_analyses_summary = "\n".join(
        f"{key.capitalize()} Analysis:\n" + "\n".join(value)
        for key, value in special_analyses.items()
    )

    prompt = (
        f"The dataset has the following properties:\n{summary}\n"
        f"Insights:\n{insights}\n"
        f"Special Analyses:\n{special_analyses_summary}\n"
        f"Image Analyses:\n{'\n'.join(image_analyses)}\n"
        f"The visualizations generated are: {', '.join(charts)}.\n"
        "Please summarize the dataset, describe the analysis performed, key findings, and any implications in Markdown format. "
        "Do not include code block delimiters like ```markdown or similar at the start or end of the Markdown text. "
        "Ensure the content is directly usable as a Markdown file without requiring edits."
    )
    return query_chat_completion(prompt)


def save_readme(content, charts):
    """Save narrative and charts as README.md."""
    with open("README.md", "w") as file:
        file.write(content)
        for chart in charts:
            file.write(f"![{chart}]({chart})\n")

# Main execution
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    dataset = sys.argv[1]

    try:
        # Load the dataset
        df = load_data(dataset)

        # Perform analysis
        summary = generic_analysis(df)
        print("Generic analysis completed.")

        # Detect feature types
        feature_types = detect_feature_types(df)

        # Perform special analyses
        special_analyses = perform_special_analyses(df, feature_types)

        # Query the LLM for additional insights
        insights = query_chat_completion(
            f"Analyze this dataset summary:\n{summary}\nProvide key insights and any suggestions for improvement."
        )
        print("LLM insights retrieved.")

        # Create visualizations
        charts = create_visualizations(df)
        print("Visualizations created.")

        # Narrate the story
        story = narrate_story(summary, insights, charts, special_analyses)
        print("Narrative created.")

        # Save README.md
        save_readme(story, charts)
        print("README.md generated.")
    except Exception as e:
        print("Error:", e)


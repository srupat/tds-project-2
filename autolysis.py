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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from dotenv import load_dotenv
import chardet

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable is not set.")

BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def query_chat_completion(prompt):
    """Send a chat prompt to the LLM and return the response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(BASE_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
    
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
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_stats": df.describe(include="all").to_dict(),
    }

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

def advanced_statistical_analysis(df):
    """Perform advanced statistical analyses."""
    numeric_df = df.select_dtypes(include=['float', 'int'])
    
    # Advanced statistical tests
    analysis = {
        'normality_tests': {},
        'statistical_significance': {}
    }
    
    # Shapiro-Wilk test for normality
    for column in numeric_df.columns:
        stat, p = stats.shapiro(numeric_df[column].dropna())
        analysis['normality_tests'][column] = {
            'statistic': stat,
            'p_value': p,
            'is_normal_distribution': p > 0.05
        }
    
    # Feature importance using mutual information
    X = numeric_df.dropna()
    feature_importance = mutual_info_classif(X, np.zeros(len(X)))
    analysis['feature_importance'] = dict(zip(X.columns, feature_importance))
    
    # Principal Component Analysis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    analysis['pca'] = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance_explained': np.cumsum(pca.explained_variance_ratio_).tolist()
    }
    
    return analysis

def create_enhanced_visualizations(df):
    """Generate more informative and accessible visualizations."""
    # Color-blind friendly palette
    color_blind_palette = sns.color_palette("colorblind")
    
    numeric_df = preprocess_data(df)
    visualization_df = preprocess_for_visualization(numeric_df)

    # Correlation Heatmap with Improved Labeling
    plt.figure(figsize=(12, 10))
    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask to remove redundant triangular portion
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", cbar_kws={'shrink': 0.8}, 
                mask=mask, square=True, linewidths=0.5, 
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Feature Correlation Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("enhanced_correlation_heatmap.png", dpi=300)
    plt.close()

    # Advanced Outlier Detection with More Context
    plt.figure(figsize=(10, 8))
    model = IsolationForest(random_state=42, contamination=0.1)
    visualization_df['outlier_score'] = model.fit_predict(visualization_df)
    
    outliers = visualization_df[visualization_df['outlier_score'] == -1]
    non_outliers = visualization_df[visualization_df['outlier_score'] == 1]
    
    plt.scatter(non_outliers.iloc[:, 0], non_outliers.iloc[:, 1], 
                label='Normal Data Points', color=color_blind_palette[0], alpha=0.7)
    plt.scatter(outliers.iloc[:, 0], outliers.iloc[:, 1], 
                label='Potential Outliers', color=color_blind_palette[3], marker='x')
    
    plt.title("Advanced Outlier Detection", fontsize=16, fontweight='bold')
    plt.xlabel(visualization_df.columns[0], fontweight='bold')
    plt.ylabel(visualization_df.columns[1], fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("advanced_outlier_detection.png", dpi=300)
    plt.close()

    # Enhanced Distribution Analysis
    plt.figure(figsize=(15, 5))
    selected_columns = visualization_df.columns[:5]
    for i, col in enumerate(selected_columns, 1):
        plt.subplot(1, len(selected_columns), i)
        sns.histplot(visualization_df[col], kde=True, color=color_blind_palette[i-1])
        plt.title(f"Distribution of {col}", fontweight='bold')
        plt.xlabel(col)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("enhanced_distributions.png", dpi=300)
    plt.close()

    return ["enhanced_correlation_heatmap.png", "advanced_outlier_detection.png", "enhanced_distributions.png"]

def narrate_comprehensive_story(summary, insights, advanced_analysis, charts, special_analyses):
    """Generate a more integrated and contextual narrative."""
    # Integrate actual results into the narrative prompt
    narrative_prompt = (
        f"Dataset Overview:\n{summary}\n\n"
        f"Initial Insights:\n{insights}\n\n"
        f"Advanced Statistical Analysis:\n"
        f"- Normality Tests: {'Normal' if all(test['is_normal_distribution'] for test in advanced_analysis['normality_tests'].values()) else 'Non-normal'} distribution detected\n"
        f"- Top Features by Importance: {', '.join(sorted(advanced_analysis['feature_importance'], key=advanced_analysis['feature_importance'].get, reverse=True)[:3])}\n"
        f"- PCA Variance Explained: {sum(advanced_analysis['pca']['explained_variance_ratio'][:2])*100:.2f}% in first two components\n\n"
        f"Special Analyses:\n{' '.join([f"{k.capitalize()}: {' '.join(v)}" for k, v in special_analyses.items()])}\n\n"
        f"Visual Insights from Charts: {', '.join(charts)}\n\n"
        "Provide a comprehensive, data-driven narrative that synthesizes these findings. "
        "Highlight key statistical insights, potential data patterns, and actionable recommendations. "
        "Use a professional, analytical tone with clear, concise language."
    )
    
    return query_chat_completion(narrative_prompt)

def save_readme(content, charts):
    """Save narrative and charts as README.md."""
    with open("README.md", "w") as file:
        file.write(content)
        for chart in charts:
            file.write(f"![{chart}]({chart})\n")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    dataset = sys.argv[1]

    try:
        # Load the dataset
        df = load_data(dataset)

        # Perform generic analysis
        summary = generic_analysis(df)
        print("Generic analysis completed.")

        # Detect feature types
        feature_types = detect_feature_types(df)

        # Perform special analyses
        special_analyses = perform_special_analyses(df, feature_types)

        # Perform advanced statistical analysis
        advanced_analysis = advanced_statistical_analysis(df)
        print("Advanced statistical analysis completed.")

        # Query the LLM for initial insights
        insights = query_chat_completion(
            f"Analyze this dataset summary:\n{summary}\n"
            f"Advanced Analysis Hints:\n{advanced_analysis}\n"
            "Provide initial insights, potential correlations, and data quality observations."
        )
        print("LLM insights retrieved.")

        # Create enhanced visualizations
        charts = create_enhanced_visualizations(df)
        print("Enhanced visualizations created.")

        # Generate comprehensive narrative
        story = narrate_comprehensive_story(
            summary, insights, advanced_analysis, charts, special_analyses
        )
        print("Comprehensive narrative created.")

        # Save README.md
        save_readme(story, charts)
        print("README.md generated with comprehensive analysis.")

    except Exception as e:
        import traceback
        print("Error:", e)
        traceback.print_exc()
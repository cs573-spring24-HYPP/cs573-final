import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


purdue_colors = LinearSegmentedColormap.from_list(
        "Purdue", [(0.812, 0.725, 0.612), (0.0, 0.0, 0.0)], N=20)


# Function for removing correlated features (in place)
def remove_correlated_features(*dfs: pd.DataFrame, threshold: float, display_matrix: bool = True) -> None:
    corr_matrix = dfs[0].corr(numeric_only=True)

    if display_matrix:
        sns.heatmap(corr_matrix, cmap=purdue_colors)
        plt.title("Features Correlation")
        plt.show()

    drop_cols = []
    # Iterate through the correlation matrix and compare correlations
    for i in range(len(corr_matrix.columns) - 1):
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print("Correlated features: ", col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    for df in dfs:
        df.drop(columns=drops, inplace=True)
    print(f"Removed Columns {drops}")


def display_confusion_matrix(y_pred, y_test, display_labels=None, title=None) -> None:
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred,
                          labels=np.unique(y_test))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=display_labels if display_labels is not None else np.unique(y_test)).plot(
        cmap=purdue_colors,
    )
    if title:
        plt.title(title)
    plt.show()

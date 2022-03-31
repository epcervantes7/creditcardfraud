import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

MODEL = joblib.load('best_gs_pipeline.pkl')
df = pd.read_csv("test.csv")
X_test = df.iloc[:,:-1]
y_test = df['target']
class_names = ['Normal','Fraud']
titles_options = [
    ("Confusion matrix, without normalization","cf", None),
    ("Normalized confusion matrix","ncf", "true"),
]
for title,fln, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        MODEL,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

#     print(title)
#     print(disp.confusion_matrix)
    plt.savefig(fln+".png",dpi=120) 

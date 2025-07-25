{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "71hiSmrQll85"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sb"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DfJ4QhbElw5t"
      },
      "outputs": [],
      "source": [
        "from sklearn import metrics\n",
        "from sklearn.metrics import classification_report\n",
        "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score\n",
        "from sklearn.metrics import roc_auc_score\n",
        "from sklearn.metrics import precision_recall_curve\n",
        "from sklearn.metrics import matthews_corrcoef\n",
        "from sklearn.model_selection import train_test_split"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "AG0BGuUvlyr1"
      },
      "outputs": [],
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.tree import ExtraTreeClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier\n",
        "from sklearn.ensemble import AdaBoostClassifier\n",
        "from sklearn.ensemble import GradientBoostingClassifier\n",
        "from xgboost import XGBClassifier\n",
        "from lightgbm import LGBMClassifier\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.ensemble import StackingClassifier\n",
        "from sklearn.model_selection import KFold\n",
        "from sklearn.model_selection import cross_val_score, cross_val_predict"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout,LSTM"
      ],
      "metadata": {
        "id": "-Bz-7hYjwj-t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fbGNjODIl1lX"
      },
      "outputs": [],
      "source": [
        "# estimator = [('RF', RandomForestClassifier(n_estimators = 200, max_depth = 5)), ('GB', GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.5, random_state = 50)),\n",
        "#              ('CAT', CatBoostClassifier(depth= 5, iterations = 35, learning_rate = 0.35)), ('ADB', AdaBoostClassifier(n_estimators = 200, learning_rate = 0.1, random_state = 50))]\n",
        "# Stacking = StackingClassifier( estimators=estimator, final_estimator= CatBoostClassifier(depth= 5, iterations = 35, learning_rate = 0.35))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OFgFIuJQOYle"
      },
      "source": [
        "**Train**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2vYkJLhWl-Ks"
      },
      "outputs": [],
      "source": [
        "df = pd.read_csv('/content/FastText+CTDT.csv')\n",
        "columns = df.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df[columns]\n",
        "Y = df[target]"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3)"
      ],
      "metadata": {
        "id": "54qCUm3mGbDZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fUUuv7DhmBE0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "064549cf-0a33-477c-99a9-86c24915516b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                                          Classifier  Accuracy       mcc  \\\n",
            "0  RandomForestClassifier(max_depth=5, n_estimato...  0.800654  0.600265   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.787582  0.574367   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.807190  0.615023   \n",
            "3  GradientBoostingClassifier(learning_rate=0.5, ...  0.823529  0.646162   \n",
            "4  AdaBoostClassifier(learning_rate=0.1, n_estima...  0.790850  0.580517   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.599631   0.805755  0.767123  0.785965      0.83125     0.767123  \n",
            "1  0.572864   0.800000  0.739726  0.768683      0.83125     0.739726  \n",
            "2  0.614379   0.784314  0.821918  0.802676      0.79375     0.821918  \n",
            "3  0.646106   0.819444  0.808219  0.813793      0.83750     0.808219  \n",
            "4  0.580067   0.792857  0.760274  0.776224      0.81875     0.760274  \n"
          ]
        }
      ],
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 200, max_depth = 5),\n",
        "          XGBClassifier(n_estimators = 200,max_depth = 5, learning_rate = 0.1),\n",
        "          LGBMClassifier(learning_rate = 0.1,max_depth = 5,random_state = 50),\n",
        "          GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.5, random_state = 50),\n",
        "          AdaBoostClassifier(n_estimators = 200, learning_rate = 0.1, random_state = 50)]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  # model.fit(xtrain, ytrain)\n",
        "  # pred = model.predict(xtest)\n",
        "  pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytrain, pred)\n",
        "  mcc = matthews_corrcoef(ytrain, pred)\n",
        "  cm1 = confusion_matrix(ytrain, pred)\n",
        "  kappa = cohen_kappa_score(ytrain, pred)\n",
        "  f1 = f1_score(ytrain, pred)\n",
        "  precision_score = precision_score(ytrain, pred)\n",
        "  recall_score = recall_score(ytrain, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"total_Metics((FT+CTDT)-CV).csv\")\n",
        "# clf = StackingClassifier( estimators=estimator, final_estimator=RandomForestClassifier(n_estimators = 200, max_depth = 5))\n",
        "# prob = clf.fit_transform(xtrain, ytrain)\n",
        "# pd.DataFrame(prob).to_csv(\"total_Metics_Probability.csv\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "docYyaZgOf4_"
      },
      "source": [
        "**Test**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kpGZDh3uOwUG",
        "outputId": "93f3f9ec-f72b-4853-c0ef-0ffd50b4eae4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Info] Number of positive: 146, number of negative: 160\n",
            "[LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000949 seconds.\n",
            "You can set `force_col_wise=true` to remove the overhead.\n",
            "[LightGBM] [Info] Total Bins 15079\n",
            "[LightGBM] [Info] Number of data points in the train set: 306, number of used features: 167\n",
            "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.477124 -> initscore=-0.091567\n",
            "[LightGBM] [Info] Start training from score -0.091567\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "                                          Classifier  Accuracy       mcc  \\\n",
            "0  (DecisionTreeClassifier(max_depth=5, max_featu...  0.871212  0.746636   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.871212  0.740003   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.901515  0.802596   \n",
            "3  ([DecisionTreeRegressor(criterion='friedman_ms...  0.886364  0.774234   \n",
            "4  (DecisionTreeClassifier(max_depth=1, random_st...  0.886364  0.790079   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.742424   0.924242  0.835616  0.877698     0.915254     0.835616  \n",
            "1  0.739917   0.888889  0.876712  0.882759     0.864407     0.876712  \n",
            "2  0.801756   0.928571  0.890411  0.909091     0.915254     0.890411  \n",
            "3  0.771994   0.926471  0.863014  0.893617     0.915254     0.863014  \n",
            "4  0.774898   0.983333  0.808219  0.887218     0.983051     0.808219  \n"
          ]
        }
      ],
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['Accuracy'] = 'Accuracy'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "# cv = KFold(n_splits=5, random_state=1, shuffle=True)\n",
        "\n",
        "# create model\n",
        "models = [RandomForestClassifier(n_estimators = 200, max_depth = 5),\n",
        "          XGBClassifier(n_estimators = 200,max_depth = 5, learning_rate = 0.1),\n",
        "          LGBMClassifier(learning_rate = 0.1,max_depth = 5,random_state = 50),\n",
        "          GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.5, random_state = 50),\n",
        "          AdaBoostClassifier(n_estimators = 200, learning_rate = 0.1, random_state = 50)]\n",
        "for model in models:\n",
        "  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "  # evaluate model\n",
        "  # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)\n",
        "  model.fit(xtrain, ytrain)\n",
        "  pred = model.predict(xtest)\n",
        "  # pred = cross_val_predict(model, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "\n",
        "  # cm1 = confusion_matrix(y, y_pred)\n",
        "  # report performance\n",
        "  Accuracy = accuracy_score(ytest, pred)\n",
        "  mcc = matthews_corrcoef(ytest, pred)\n",
        "  cm1 = confusion_matrix(ytest, pred)\n",
        "  kappa = cohen_kappa_score(ytest, pred)\n",
        "  f1 = f1_score(ytest, pred)\n",
        "  precision_score = precision_score(ytest, pred)\n",
        "  recall_score = recall_score(ytest, pred)\n",
        "  sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "  specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "  # y_pred = np.argmax(y_pred, axis=0)\n",
        "  # auc = roc_auc_score(y, y_pred, multi_class='ovr')\n",
        "  total_Metics.loc[len(total_Metics.index)] = [model,Accuracy, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]\n",
        "\n",
        "print(total_Metics)\n",
        "total_Metics.to_csv(\"total_Metics((FT+CTDT)-TS)).csv\")\n",
        "# clf = StackingClassifier( estimators=estimator, final_estimator=RandomForestClassifier(n_estimators = 200, max_depth = 5))\n",
        "# prob = clf.fit_transform(xtest, ytest)\n",
        "# pd.DataFrame(prob).to_csv(\"total_Metics_Probability(LSA+PAAC).csv\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **CNN**"
      ],
      "metadata": {
        "id": "5zeJ0H9I373c"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('/content/Res-FastText+CTDT.csv')\n",
        "columns = df.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df[columns]\n",
        "Y = df[target]"
      ],
      "metadata": {
        "id": "mMLDGLBRAwfm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Train**"
      ],
      "metadata": {
        "id": "3BLVQrcUB6rA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3)"
      ],
      "metadata": {
        "id": "Y3gZHPKDA0te"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "xtrain = xtrain.to_numpy()\n",
        "ytrain = ytrain.to_numpy()\n",
        "xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)\n",
        "# xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)"
      ],
      "metadata": {
        "id": "cv0C6ZF639bb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(xtrain):\n",
        "    X_train, X_val = xtrain[train_index], xtrain[val_index]\n",
        "    y_train, y_val = ytrain[train_index], ytrain[val_index]"
      ],
      "metadata": {
        "id": "MMwz6ZaP4NAN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()"
      ],
      "metadata": {
        "id": "gAB9oIwJ4cie"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=2, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=2, activation='relu'))"
      ],
      "metadata": {
        "id": "rGhc3hTl4fYT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ],
      "metadata": {
        "id": "2gy_kUsg4mxD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "ldx-dq454sdt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(128, activation='relu'))\n",
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "HhCvyAnq4uXb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "bPKlJAE04wWN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(X_train, y_train, epochs = 40, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hif7w0dG4yh0",
        "outputId": "8de5f45f-f66d-4d12-ea78-22d180776e5b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "4/4 [==============================] - 2s 168ms/step - loss: 0.6889 - accuracy: 0.4980\n",
            "Epoch 2/40\n",
            "4/4 [==============================] - 1s 174ms/step - loss: 0.6596 - accuracy: 0.5918\n",
            "Epoch 3/40\n",
            "4/4 [==============================] - 1s 186ms/step - loss: 0.6086 - accuracy: 0.7429\n",
            "Epoch 4/40\n",
            "4/4 [==============================] - 1s 134ms/step - loss: 0.5300 - accuracy: 0.7878\n",
            "Epoch 5/40\n",
            "4/4 [==============================] - 0s 103ms/step - loss: 0.4788 - accuracy: 0.7878\n",
            "Epoch 6/40\n",
            "4/4 [==============================] - 0s 101ms/step - loss: 0.4241 - accuracy: 0.8082\n",
            "Epoch 7/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 0.4287 - accuracy: 0.7633\n",
            "Epoch 8/40\n",
            "4/4 [==============================] - 0s 101ms/step - loss: 0.3377 - accuracy: 0.8612\n",
            "Epoch 9/40\n",
            "4/4 [==============================] - 0s 104ms/step - loss: 0.3256 - accuracy: 0.8694\n",
            "Epoch 10/40\n",
            "4/4 [==============================] - 0s 103ms/step - loss: 0.3052 - accuracy: 0.8694\n",
            "Epoch 11/40\n",
            "4/4 [==============================] - 0s 100ms/step - loss: 0.3224 - accuracy: 0.8449\n",
            "Epoch 12/40\n",
            "4/4 [==============================] - 0s 100ms/step - loss: 0.3102 - accuracy: 0.8857\n",
            "Epoch 13/40\n",
            "4/4 [==============================] - 0s 107ms/step - loss: 0.2721 - accuracy: 0.8816\n",
            "Epoch 14/40\n",
            "4/4 [==============================] - 0s 105ms/step - loss: 0.2538 - accuracy: 0.9061\n",
            "Epoch 15/40\n",
            "4/4 [==============================] - 0s 101ms/step - loss: 0.2644 - accuracy: 0.8898\n",
            "Epoch 16/40\n",
            "4/4 [==============================] - 0s 110ms/step - loss: 0.2649 - accuracy: 0.8776\n",
            "Epoch 17/40\n",
            "4/4 [==============================] - 0s 105ms/step - loss: 0.2304 - accuracy: 0.9102\n",
            "Epoch 18/40\n",
            "4/4 [==============================] - 0s 98ms/step - loss: 0.2250 - accuracy: 0.9020\n",
            "Epoch 19/40\n",
            "4/4 [==============================] - 0s 112ms/step - loss: 0.2390 - accuracy: 0.9061\n",
            "Epoch 20/40\n",
            "4/4 [==============================] - 0s 105ms/step - loss: 0.2365 - accuracy: 0.9102\n",
            "Epoch 21/40\n",
            "4/4 [==============================] - 0s 101ms/step - loss: 0.2469 - accuracy: 0.8980\n",
            "Epoch 22/40\n",
            "4/4 [==============================] - 0s 100ms/step - loss: 0.2135 - accuracy: 0.9306\n",
            "Epoch 23/40\n",
            "4/4 [==============================] - 0s 100ms/step - loss: 0.1923 - accuracy: 0.9306\n",
            "Epoch 24/40\n",
            "4/4 [==============================] - 0s 100ms/step - loss: 0.1654 - accuracy: 0.9469\n",
            "Epoch 25/40\n",
            "4/4 [==============================] - 0s 96ms/step - loss: 0.1520 - accuracy: 0.9429\n",
            "Epoch 26/40\n",
            "4/4 [==============================] - 0s 102ms/step - loss: 0.1531 - accuracy: 0.9469\n",
            "Epoch 27/40\n",
            "4/4 [==============================] - 0s 103ms/step - loss: 0.1400 - accuracy: 0.9469\n",
            "Epoch 28/40\n",
            "4/4 [==============================] - 1s 148ms/step - loss: 0.1593 - accuracy: 0.9510\n",
            "Epoch 29/40\n",
            "4/4 [==============================] - 1s 180ms/step - loss: 0.1525 - accuracy: 0.9429\n",
            "Epoch 30/40\n",
            "4/4 [==============================] - 1s 172ms/step - loss: 0.1389 - accuracy: 0.9510\n",
            "Epoch 31/40\n",
            "4/4 [==============================] - 1s 181ms/step - loss: 0.1210 - accuracy: 0.9510\n",
            "Epoch 32/40\n",
            "4/4 [==============================] - 1s 187ms/step - loss: 0.1159 - accuracy: 0.9673\n",
            "Epoch 33/40\n",
            "4/4 [==============================] - 1s 119ms/step - loss: 0.1037 - accuracy: 0.9633\n",
            "Epoch 34/40\n",
            "4/4 [==============================] - 0s 98ms/step - loss: 0.0977 - accuracy: 0.9673\n",
            "Epoch 35/40\n",
            "4/4 [==============================] - 0s 110ms/step - loss: 0.0853 - accuracy: 0.9796\n",
            "Epoch 36/40\n",
            "4/4 [==============================] - 0s 101ms/step - loss: 0.0798 - accuracy: 0.9837\n",
            "Epoch 37/40\n",
            "4/4 [==============================] - 0s 102ms/step - loss: 0.0764 - accuracy: 0.9837\n",
            "Epoch 38/40\n",
            "4/4 [==============================] - 0s 109ms/step - loss: 0.0691 - accuracy: 0.9837\n",
            "Epoch 39/40\n",
            "4/4 [==============================] - 0s 99ms/step - loss: 0.0703 - accuracy: 0.9878\n",
            "Epoch 40/40\n",
            "4/4 [==============================] - 0s 105ms/step - loss: 0.0715 - accuracy: 0.9755\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x797214753f40>"
            ]
          },
          "metadata": {},
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QNFWF72141JP",
        "outputId": "7b1df7d4-2a63-4fab-9ba5-0b2de1dd2ce8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2/2 [==============================] - 0s 17ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(y_val, y_pred_classes), f1_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0Jm6bHLH5MA1",
        "outputId": "45ad44b8-7002-47a5-b95c-38fbd1eec964"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9508196721311475, 0.9538461538461539, 0.901241230437129, 0.901729964992645)"
            ]
          },
          "metadata": {},
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "specificity, sensitivity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aCX03HuC5OWG",
        "outputId": "e847c67b-3b10-4731-afea-422e3e18291b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9310344827586207, 0.96875)"
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Test**"
      ],
      "metadata": {
        "id": "k1XklLHsCBy_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3)"
      ],
      "metadata": {
        "id": "0bzm03V7CJ2c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ],
      "metadata": {
        "id": "2zF9-4E7CDI9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ],
      "metadata": {
        "id": "O0dDhydTCFpM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn = Sequential()\n",
        "cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=2, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=2, activation='relu'))"
      ],
      "metadata": {
        "id": "OpVNuOpMCHos"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(MaxPool1D(pool_size=4))"
      ],
      "metadata": {
        "id": "6ocTvVGtCV38"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Flatten())"
      ],
      "metadata": {
        "id": "PAHg1CwVCYzM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.add(Dense(128, activation='relu'))\n",
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "id": "V3wk12xICcAs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "metadata": {
        "id": "bOg1QfMpCfZE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn.fit(xtrain, ytrain, epochs = 40, batch_size= 64)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LmPqBaaWCiJc",
        "outputId": "e9edda36-4641-482f-c69c-75be761801cc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "5/5 [==============================] - 2s 96ms/step - loss: 0.6883 - accuracy: 0.5359\n",
            "Epoch 2/40\n",
            "5/5 [==============================] - 0s 92ms/step - loss: 0.6687 - accuracy: 0.5327\n",
            "Epoch 3/40\n",
            "5/5 [==============================] - 1s 100ms/step - loss: 0.6279 - accuracy: 0.6013\n",
            "Epoch 4/40\n",
            "5/5 [==============================] - 0s 94ms/step - loss: 0.5791 - accuracy: 0.7418\n",
            "Epoch 5/40\n",
            "5/5 [==============================] - 0s 98ms/step - loss: 0.5467 - accuracy: 0.7386\n",
            "Epoch 6/40\n",
            "5/5 [==============================] - 0s 98ms/step - loss: 0.4914 - accuracy: 0.7810\n",
            "Epoch 7/40\n",
            "5/5 [==============================] - 1s 103ms/step - loss: 0.4657 - accuracy: 0.7876\n",
            "Epoch 8/40\n",
            "5/5 [==============================] - 0s 89ms/step - loss: 0.4456 - accuracy: 0.7647\n",
            "Epoch 9/40\n",
            "5/5 [==============================] - 0s 97ms/step - loss: 0.4339 - accuracy: 0.8170\n",
            "Epoch 10/40\n",
            "5/5 [==============================] - 1s 104ms/step - loss: 0.3947 - accuracy: 0.8301\n",
            "Epoch 11/40\n",
            "5/5 [==============================] - 1s 159ms/step - loss: 0.3665 - accuracy: 0.8497\n",
            "Epoch 12/40\n",
            "5/5 [==============================] - 1s 164ms/step - loss: 0.3450 - accuracy: 0.8595\n",
            "Epoch 13/40\n",
            "5/5 [==============================] - 1s 161ms/step - loss: 0.3214 - accuracy: 0.8627\n",
            "Epoch 14/40\n",
            "5/5 [==============================] - 1s 155ms/step - loss: 0.3155 - accuracy: 0.8791\n",
            "Epoch 15/40\n",
            "5/5 [==============================] - 0s 93ms/step - loss: 0.2898 - accuracy: 0.9052\n",
            "Epoch 16/40\n",
            "5/5 [==============================] - 0s 88ms/step - loss: 0.2914 - accuracy: 0.8824\n",
            "Epoch 17/40\n",
            "5/5 [==============================] - 0s 95ms/step - loss: 0.2865 - accuracy: 0.8922\n",
            "Epoch 18/40\n",
            "5/5 [==============================] - 0s 86ms/step - loss: 0.2662 - accuracy: 0.9118\n",
            "Epoch 19/40\n",
            "5/5 [==============================] - 0s 92ms/step - loss: 0.2418 - accuracy: 0.9216\n",
            "Epoch 20/40\n",
            "5/5 [==============================] - 0s 92ms/step - loss: 0.2345 - accuracy: 0.9150\n",
            "Epoch 21/40\n",
            "5/5 [==============================] - 0s 91ms/step - loss: 0.2429 - accuracy: 0.8987\n",
            "Epoch 22/40\n",
            "5/5 [==============================] - 0s 90ms/step - loss: 0.2507 - accuracy: 0.9118\n",
            "Epoch 23/40\n",
            "5/5 [==============================] - 0s 90ms/step - loss: 0.2621 - accuracy: 0.8889\n",
            "Epoch 24/40\n",
            "5/5 [==============================] - 0s 93ms/step - loss: 0.2295 - accuracy: 0.9216\n",
            "Epoch 25/40\n",
            "5/5 [==============================] - 0s 94ms/step - loss: 0.2099 - accuracy: 0.9412\n",
            "Epoch 26/40\n",
            "5/5 [==============================] - 0s 96ms/step - loss: 0.1933 - accuracy: 0.9412\n",
            "Epoch 27/40\n",
            "5/5 [==============================] - 0s 90ms/step - loss: 0.1822 - accuracy: 0.9379\n",
            "Epoch 28/40\n",
            "5/5 [==============================] - 0s 91ms/step - loss: 0.1861 - accuracy: 0.9346\n",
            "Epoch 29/40\n",
            "5/5 [==============================] - 0s 90ms/step - loss: 0.1757 - accuracy: 0.9379\n",
            "Epoch 30/40\n",
            "5/5 [==============================] - 0s 92ms/step - loss: 0.1648 - accuracy: 0.9412\n",
            "Epoch 31/40\n",
            "5/5 [==============================] - 0s 89ms/step - loss: 0.1452 - accuracy: 0.9444\n",
            "Epoch 32/40\n",
            "5/5 [==============================] - 0s 90ms/step - loss: 0.1544 - accuracy: 0.9510\n",
            "Epoch 33/40\n",
            "5/5 [==============================] - 0s 93ms/step - loss: 0.1317 - accuracy: 0.9608\n",
            "Epoch 34/40\n",
            "5/5 [==============================] - 0s 86ms/step - loss: 0.1403 - accuracy: 0.9542\n",
            "Epoch 35/40\n",
            "5/5 [==============================] - 0s 90ms/step - loss: 0.1639 - accuracy: 0.9444\n",
            "Epoch 36/40\n",
            "5/5 [==============================] - 1s 108ms/step - loss: 0.1202 - accuracy: 0.9641\n",
            "Epoch 37/40\n",
            "5/5 [==============================] - 1s 158ms/step - loss: 0.1034 - accuracy: 0.9771\n",
            "Epoch 38/40\n",
            "5/5 [==============================] - 1s 159ms/step - loss: 0.0982 - accuracy: 0.9673\n",
            "Epoch 39/40\n",
            "5/5 [==============================] - 1s 162ms/step - loss: 0.0897 - accuracy: 0.9869\n",
            "Epoch 40/40\n",
            "5/5 [==============================] - 1s 149ms/step - loss: 0.0842 - accuracy: 0.9869\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x797215bee620>"
            ]
          },
          "metadata": {},
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = cnn.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4NAgGEaVCmCE",
        "outputId": "afacc4b9-0a60-4f45-e185-601032e1f688"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5/5 [==============================] - 0s 14ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QGek9g4PCoT1",
        "outputId": "f96dfff4-7562-4d48-cf81-d89218b54724"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9318181818181818,\n",
              " 0.9343065693430657,\n",
              " 0.8638239339752407,\n",
              " 0.8687102404263812)"
            ]
          },
          "metadata": {},
          "execution_count": 59
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "specificity, sensitivity"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GcTwK6-kCr3V",
        "outputId": "ddcb75ca-cc63-464e-f82f-19a2da37800a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8805970149253731, 0.9846153846153847)"
            ]
          },
          "metadata": {},
          "execution_count": 60
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

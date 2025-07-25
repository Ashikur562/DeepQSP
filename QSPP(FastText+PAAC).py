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
        "df = pd.read_csv('/content/FastText+PAAC.csv')\n",
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
        "outputId": "09a9b795-73ad-462d-f170-14154b66976b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                                          Classifier  Accuracy       mcc  \\\n",
            "0  RandomForestClassifier(max_depth=5, n_estimato...  0.856209  0.716010   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.869281  0.738234   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.852941  0.706084   \n",
            "3  GradientBoostingClassifier(learning_rate=0.5, ...  0.866013  0.732257   \n",
            "4  AdaBoostClassifier(learning_rate=0.1, n_estima...  0.843137  0.688024   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.713019   0.818750  0.897260  0.856209      0.81875     0.897260  \n",
            "1  0.738171   0.858108  0.869863  0.863946      0.86875     0.869863  \n",
            "2  0.705706   0.834437  0.863014  0.848485      0.84375     0.863014  \n",
            "3  0.731866   0.847682  0.876712  0.861953      0.85625     0.876712  \n",
            "4  0.686556   0.814103  0.869863  0.841060      0.81875     0.869863  \n"
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
        "total_Metics.to_csv(\"total_Metics((FT+PAAC)-CV).csv\")\n",
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
        "outputId": "1c5164fb-62be-4864-d666-03f865ec6dd3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Info] Number of positive: 146, number of negative: 160\n",
            "[LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001323 seconds.\n",
            "You can set `force_col_wise=true` to remove the overhead.\n",
            "[LightGBM] [Info] Total Bins 14426\n",
            "[LightGBM] [Info] Number of data points in the train set: 306, number of used features: 150\n",
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
            "0  (DecisionTreeClassifier(max_depth=5, max_featu...  0.825758  0.646514   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.825758  0.648198   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.818182  0.631022   \n",
            "3  ([DecisionTreeRegressor(criterion='friedman_ms...  0.848485  0.694839   \n",
            "4  (DecisionTreeClassifier(max_depth=1, random_st...  0.833333  0.662088   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.645824   0.828947  0.863014  0.845638     0.779661     0.863014  \n",
            "1  0.648122   0.847222  0.835616  0.841379     0.813559     0.835616  \n",
            "2  0.629820   0.818182  0.863014  0.840000     0.762712     0.863014  \n",
            "3  0.694515   0.873239  0.849315  0.861111     0.847458     0.849315  \n",
            "4  0.661775   0.840000  0.863014  0.851351     0.796610     0.863014  \n"
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
        "total_Metics.to_csv(\"total_Metics((FT+PAAC)-TS)).csv\")\n",
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
        "df = pd.read_csv('/content/Res-FastText+PAAC.csv')\n",
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
        "outputId": "b55d4c40-a2da-4637-fb7e-09ccbbaf333f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "4/4 [==============================] - 2s 90ms/step - loss: 0.6507 - accuracy: 0.4735\n",
            "Epoch 2/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 0.5500 - accuracy: 0.7020\n",
            "Epoch 3/40\n",
            "4/4 [==============================] - 0s 95ms/step - loss: 0.4292 - accuracy: 0.8531\n",
            "Epoch 4/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 0.3579 - accuracy: 0.8571\n",
            "Epoch 5/40\n",
            "4/4 [==============================] - 1s 243ms/step - loss: 0.3305 - accuracy: 0.8939\n",
            "Epoch 6/40\n",
            "4/4 [==============================] - 1s 101ms/step - loss: 0.2836 - accuracy: 0.8857\n",
            "Epoch 7/40\n",
            "4/4 [==============================] - 1s 231ms/step - loss: 0.2601 - accuracy: 0.9102\n",
            "Epoch 8/40\n",
            "4/4 [==============================] - 1s 100ms/step - loss: 0.2410 - accuracy: 0.9143\n",
            "Epoch 9/40\n",
            "4/4 [==============================] - 0s 98ms/step - loss: 0.2296 - accuracy: 0.9184\n",
            "Epoch 10/40\n",
            "4/4 [==============================] - 0s 104ms/step - loss: 0.2115 - accuracy: 0.9265\n",
            "Epoch 11/40\n",
            "4/4 [==============================] - 0s 106ms/step - loss: 0.1819 - accuracy: 0.9510\n",
            "Epoch 12/40\n",
            "4/4 [==============================] - 0s 98ms/step - loss: 0.1736 - accuracy: 0.9551\n",
            "Epoch 13/40\n",
            "4/4 [==============================] - 0s 99ms/step - loss: 0.1533 - accuracy: 0.9510\n",
            "Epoch 14/40\n",
            "4/4 [==============================] - 0s 87ms/step - loss: 0.1402 - accuracy: 0.9592\n",
            "Epoch 15/40\n",
            "4/4 [==============================] - 0s 91ms/step - loss: 0.1243 - accuracy: 0.9755\n",
            "Epoch 16/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 0.1090 - accuracy: 0.9755\n",
            "Epoch 17/40\n",
            "4/4 [==============================] - 0s 88ms/step - loss: 0.0990 - accuracy: 0.9796\n",
            "Epoch 18/40\n",
            "4/4 [==============================] - 0s 109ms/step - loss: 0.0867 - accuracy: 0.9755\n",
            "Epoch 19/40\n",
            "4/4 [==============================] - 1s 172ms/step - loss: 0.0734 - accuracy: 0.9837\n",
            "Epoch 20/40\n",
            "4/4 [==============================] - 1s 179ms/step - loss: 0.0724 - accuracy: 0.9796\n",
            "Epoch 21/40\n",
            "4/4 [==============================] - 1s 178ms/step - loss: 0.0855 - accuracy: 0.9755\n",
            "Epoch 22/40\n",
            "4/4 [==============================] - 1s 150ms/step - loss: 0.1021 - accuracy: 0.9673\n",
            "Epoch 23/40\n",
            "4/4 [==============================] - 1s 125ms/step - loss: 0.0564 - accuracy: 0.9878\n",
            "Epoch 24/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 0.0507 - accuracy: 0.9878\n",
            "Epoch 25/40\n",
            "4/4 [==============================] - 0s 98ms/step - loss: 0.0495 - accuracy: 0.9918\n",
            "Epoch 26/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 0.0432 - accuracy: 0.9918\n",
            "Epoch 27/40\n",
            "4/4 [==============================] - 0s 91ms/step - loss: 0.0309 - accuracy: 0.9918\n",
            "Epoch 28/40\n",
            "4/4 [==============================] - 0s 93ms/step - loss: 0.0273 - accuracy: 0.9918\n",
            "Epoch 29/40\n",
            "4/4 [==============================] - 0s 104ms/step - loss: 0.0260 - accuracy: 0.9918\n",
            "Epoch 30/40\n",
            "4/4 [==============================] - 0s 103ms/step - loss: 0.0210 - accuracy: 0.9918\n",
            "Epoch 31/40\n",
            "4/4 [==============================] - 0s 96ms/step - loss: 0.0221 - accuracy: 0.9959\n",
            "Epoch 32/40\n",
            "4/4 [==============================] - 0s 100ms/step - loss: 0.0165 - accuracy: 0.9959\n",
            "Epoch 33/40\n",
            "4/4 [==============================] - 0s 102ms/step - loss: 0.0148 - accuracy: 0.9959\n",
            "Epoch 34/40\n",
            "4/4 [==============================] - 0s 101ms/step - loss: 0.0125 - accuracy: 0.9959\n",
            "Epoch 35/40\n",
            "4/4 [==============================] - 0s 102ms/step - loss: 0.0096 - accuracy: 1.0000\n",
            "Epoch 36/40\n",
            "4/4 [==============================] - 0s 96ms/step - loss: 0.0111 - accuracy: 0.9959\n",
            "Epoch 37/40\n",
            "4/4 [==============================] - 0s 88ms/step - loss: 0.0076 - accuracy: 1.0000\n",
            "Epoch 38/40\n",
            "4/4 [==============================] - 0s 91ms/step - loss: 0.0062 - accuracy: 1.0000\n",
            "Epoch 39/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 0.0067 - accuracy: 0.9959\n",
            "Epoch 40/40\n",
            "4/4 [==============================] - 0s 111ms/step - loss: 0.0068 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7e4da0940be0>"
            ]
          },
          "metadata": {},
          "execution_count": 46
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
        "outputId": "8e0ad6a9-9a01-4334-b7de-10ed0ac8163c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2/2 [==============================] - 0s 30ms/step\n"
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
        "outputId": "7db3a273-5e11-4f32-dec0-09b36a08951c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9344262295081968,\n",
              " 0.9310344827586207,\n",
              " 0.8685344827586207,\n",
              " 0.8685344827586207)"
            ]
          },
          "metadata": {},
          "execution_count": 48
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
        "outputId": "d51591e3-16be-4816-a43d-d31efcdb74ee"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9375, 0.9310344827586207)"
            ]
          },
          "metadata": {},
          "execution_count": 49
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
        "outputId": "21216b21-fc85-42b1-ee9d-cbf71cd3db4e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "5/5 [==============================] - 2s 86ms/step - loss: 0.6560 - accuracy: 0.4902\n",
            "Epoch 2/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.5168 - accuracy: 0.8268\n",
            "Epoch 3/40\n",
            "5/5 [==============================] - 0s 81ms/step - loss: 0.4143 - accuracy: 0.8235\n",
            "Epoch 4/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.3283 - accuracy: 0.8889\n",
            "Epoch 5/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.3190 - accuracy: 0.8954\n",
            "Epoch 6/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.3017 - accuracy: 0.8987\n",
            "Epoch 7/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.2889 - accuracy: 0.8856\n",
            "Epoch 8/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.3070 - accuracy: 0.8889\n",
            "Epoch 9/40\n",
            "5/5 [==============================] - 0s 90ms/step - loss: 0.2664 - accuracy: 0.8987\n",
            "Epoch 10/40\n",
            "5/5 [==============================] - 0s 88ms/step - loss: 0.2537 - accuracy: 0.9150\n",
            "Epoch 11/40\n",
            "5/5 [==============================] - 1s 106ms/step - loss: 0.2422 - accuracy: 0.9085\n",
            "Epoch 12/40\n",
            "5/5 [==============================] - 1s 148ms/step - loss: 0.2186 - accuracy: 0.9150\n",
            "Epoch 13/40\n",
            "5/5 [==============================] - 1s 138ms/step - loss: 0.1972 - accuracy: 0.9379\n",
            "Epoch 14/40\n",
            "5/5 [==============================] - 1s 140ms/step - loss: 0.2060 - accuracy: 0.9248\n",
            "Epoch 15/40\n",
            "5/5 [==============================] - 1s 146ms/step - loss: 0.1846 - accuracy: 0.9444\n",
            "Epoch 16/40\n",
            "5/5 [==============================] - 1s 98ms/step - loss: 0.1917 - accuracy: 0.9412\n",
            "Epoch 17/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.1858 - accuracy: 0.9412\n",
            "Epoch 18/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.1424 - accuracy: 0.9542\n",
            "Epoch 19/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.1314 - accuracy: 0.9575\n",
            "Epoch 20/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 0.1169 - accuracy: 0.9575\n",
            "Epoch 21/40\n",
            "5/5 [==============================] - 0s 79ms/step - loss: 0.1022 - accuracy: 0.9739\n",
            "Epoch 22/40\n",
            "5/5 [==============================] - 0s 91ms/step - loss: 0.0906 - accuracy: 0.9771\n",
            "Epoch 23/40\n",
            "5/5 [==============================] - 0s 90ms/step - loss: 0.0793 - accuracy: 0.9804\n",
            "Epoch 24/40\n",
            "5/5 [==============================] - 0s 87ms/step - loss: 0.0732 - accuracy: 0.9771\n",
            "Epoch 25/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 0.0702 - accuracy: 0.9771\n",
            "Epoch 26/40\n",
            "5/5 [==============================] - 0s 81ms/step - loss: 0.0554 - accuracy: 0.9902\n",
            "Epoch 27/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 0.0515 - accuracy: 0.9869\n",
            "Epoch 28/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 0.0434 - accuracy: 0.9902\n",
            "Epoch 29/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 0.0359 - accuracy: 0.9902\n",
            "Epoch 30/40\n",
            "5/5 [==============================] - 0s 81ms/step - loss: 0.0302 - accuracy: 0.9902\n",
            "Epoch 31/40\n",
            "5/5 [==============================] - 0s 78ms/step - loss: 0.0218 - accuracy: 0.9967\n",
            "Epoch 32/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 0.0187 - accuracy: 0.9967\n",
            "Epoch 33/40\n",
            "5/5 [==============================] - 0s 79ms/step - loss: 0.0151 - accuracy: 1.0000\n",
            "Epoch 34/40\n",
            "5/5 [==============================] - 0s 78ms/step - loss: 0.0167 - accuracy: 0.9967\n",
            "Epoch 35/40\n",
            "5/5 [==============================] - 0s 79ms/step - loss: 0.0124 - accuracy: 1.0000\n",
            "Epoch 36/40\n",
            "5/5 [==============================] - 0s 81ms/step - loss: 0.0080 - accuracy: 1.0000\n",
            "Epoch 37/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 0.0062 - accuracy: 1.0000\n",
            "Epoch 38/40\n",
            "5/5 [==============================] - 0s 79ms/step - loss: 0.0053 - accuracy: 1.0000\n",
            "Epoch 39/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.0043 - accuracy: 1.0000\n",
            "Epoch 40/40\n",
            "5/5 [==============================] - 1s 127ms/step - loss: 0.0040 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7e4d9fd61090>"
            ]
          },
          "metadata": {},
          "execution_count": 58
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
        "outputId": "5cf26761-d2e3-404b-bb0a-1fea112b0119"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5/5 [==============================] - 0s 12ms/step\n"
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
        "outputId": "04e53abb-d730-4afe-fdf3-a5d7aaf8d284"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9696969696969697,\n",
              " 0.9714285714285714,\n",
              " 0.9391845196959225,\n",
              " 0.9396190204353759)"
            ]
          },
          "metadata": {},
          "execution_count": 60
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
        "outputId": "00da9ad5-c6c4-400e-9716-998871448448"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9523809523809523, 0.9855072463768116)"
            ]
          },
          "metadata": {},
          "execution_count": 61
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

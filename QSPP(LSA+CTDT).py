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
        "df = pd.read_csv('/content/LSA+CTDT.csv')\n",
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
        "outputId": "5a64a938-77a2-4d5e-90b5-c6f4809af728"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                                          Classifier  Accuracy       mcc  \\\n",
            "0  RandomForestClassifier(max_depth=5, n_estimato...  0.795455  0.590403   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.759740  0.519875   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.775974  0.551713   \n",
            "3  GradientBoostingClassifier(learning_rate=0.5, ...  0.788961  0.577185   \n",
            "4  AdaBoostClassifier(learning_rate=0.1, n_estima...  0.740260  0.481432   \n",
            "\n",
            "      Kappa  precision  recall        f1  sensitivity  specificity  \n",
            "0  0.590391   0.805031   0.800  0.802508     0.790541        0.800  \n",
            "1  0.519481   0.779221   0.750  0.764331     0.770270        0.750  \n",
            "2  0.551608   0.789809   0.775  0.782334     0.777027        0.775  \n",
            "3  0.577173   0.795031   0.800  0.797508     0.777027        0.800  \n",
            "4  0.480782   0.763158   0.725  0.743590     0.756757        0.725  \n"
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
        "total_Metics.to_csv(\"total_Metics((LSA+CTDT)-CV).csv\")\n",
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
        "outputId": "ab851868-41b4-48c7-c033-4572c8f7350d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Info] Number of positive: 160, number of negative: 148\n",
            "[LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000717 seconds.\n",
            "You can set `force_col_wise=true` to remove the overhead.\n",
            "[LightGBM] [Info] Total Bins 11103\n",
            "[LightGBM] [Info] Number of data points in the train set: 308, number of used features: 128\n",
            "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.519481 -> initscore=0.077962\n",
            "[LightGBM] [Info] Start training from score 0.077962\n",
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
            "0  (DecisionTreeClassifier(max_depth=5, max_featu...  0.825758  0.649174   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.795455  0.592051   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.825758  0.652916   \n",
            "3  ([DecisionTreeRegressor(criterion='friedman_ms...  0.818182  0.636536   \n",
            "4  (DecisionTreeClassifier(max_depth=1, random_st...  0.803030  0.602778   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.649098   0.803279  0.816667  0.809917     0.833333     0.816667  \n",
            "1  0.590345   0.753846  0.816667  0.784000     0.777778     0.816667  \n",
            "2  0.651034   0.784615  0.850000  0.816000     0.805556     0.850000  \n",
            "3  0.635359   0.781250  0.833333  0.806452     0.805556     0.833333  \n",
            "4  0.602778   0.783333  0.783333  0.783333     0.819444     0.783333  \n"
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
        "total_Metics.to_csv(\"total_Metics((LSA+CTDT)-TS)).csv\")\n",
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
        "df = pd.read_csv('/content/Res-LSA+CTDT.csv')\n",
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
        "outputId": "ebaf783d-f16c-45ed-b3da-0c06a984995d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "4/4 [==============================] - 2s 81ms/step - loss: 0.6984 - accuracy: 0.4777\n",
            "Epoch 2/40\n",
            "4/4 [==============================] - 0s 87ms/step - loss: 0.6841 - accuracy: 0.5587\n",
            "Epoch 3/40\n",
            "4/4 [==============================] - 0s 85ms/step - loss: 0.6747 - accuracy: 0.7652\n",
            "Epoch 4/40\n",
            "4/4 [==============================] - 0s 83ms/step - loss: 0.6402 - accuracy: 0.7287\n",
            "Epoch 5/40\n",
            "4/4 [==============================] - 0s 90ms/step - loss: 0.5889 - accuracy: 0.7611\n",
            "Epoch 6/40\n",
            "4/4 [==============================] - 0s 83ms/step - loss: 0.5475 - accuracy: 0.7206\n",
            "Epoch 7/40\n",
            "4/4 [==============================] - 0s 85ms/step - loss: 0.5226 - accuracy: 0.7328\n",
            "Epoch 8/40\n",
            "4/4 [==============================] - 0s 86ms/step - loss: 0.4698 - accuracy: 0.7854\n",
            "Epoch 9/40\n",
            "4/4 [==============================] - 0s 86ms/step - loss: 0.4428 - accuracy: 0.7814\n",
            "Epoch 10/40\n",
            "4/4 [==============================] - 0s 84ms/step - loss: 0.4222 - accuracy: 0.8097\n",
            "Epoch 11/40\n",
            "4/4 [==============================] - 0s 85ms/step - loss: 0.3837 - accuracy: 0.8340\n",
            "Epoch 12/40\n",
            "4/4 [==============================] - 0s 84ms/step - loss: 0.3645 - accuracy: 0.8583\n",
            "Epoch 13/40\n",
            "4/4 [==============================] - 0s 85ms/step - loss: 0.3473 - accuracy: 0.8664\n",
            "Epoch 14/40\n",
            "4/4 [==============================] - 0s 84ms/step - loss: 0.3115 - accuracy: 0.8421\n",
            "Epoch 15/40\n",
            "4/4 [==============================] - 0s 82ms/step - loss: 0.2644 - accuracy: 0.8947\n",
            "Epoch 16/40\n",
            "4/4 [==============================] - 0s 80ms/step - loss: 0.2416 - accuracy: 0.9150\n",
            "Epoch 17/40\n",
            "4/4 [==============================] - 0s 86ms/step - loss: 0.2089 - accuracy: 0.9352\n",
            "Epoch 18/40\n",
            "4/4 [==============================] - 0s 84ms/step - loss: 0.1892 - accuracy: 0.9352\n",
            "Epoch 19/40\n",
            "4/4 [==============================] - 0s 87ms/step - loss: 0.1650 - accuracy: 0.9312\n",
            "Epoch 20/40\n",
            "4/4 [==============================] - 1s 145ms/step - loss: 0.1668 - accuracy: 0.9393\n",
            "Epoch 21/40\n",
            "4/4 [==============================] - 1s 148ms/step - loss: 0.0957 - accuracy: 0.9757\n",
            "Epoch 22/40\n",
            "4/4 [==============================] - 1s 139ms/step - loss: 0.0894 - accuracy: 0.9717\n",
            "Epoch 23/40\n",
            "4/4 [==============================] - 1s 140ms/step - loss: 0.0917 - accuracy: 0.9798\n",
            "Epoch 24/40\n",
            "4/4 [==============================] - 1s 138ms/step - loss: 0.0703 - accuracy: 0.9879\n",
            "Epoch 25/40\n",
            "4/4 [==============================] - 1s 143ms/step - loss: 0.0544 - accuracy: 0.9838\n",
            "Epoch 26/40\n",
            "4/4 [==============================] - 0s 89ms/step - loss: 0.0284 - accuracy: 1.0000\n",
            "Epoch 27/40\n",
            "4/4 [==============================] - 0s 83ms/step - loss: 0.0210 - accuracy: 1.0000\n",
            "Epoch 28/40\n",
            "4/4 [==============================] - 0s 89ms/step - loss: 0.0154 - accuracy: 1.0000\n",
            "Epoch 29/40\n",
            "4/4 [==============================] - 0s 79ms/step - loss: 0.0122 - accuracy: 1.0000\n",
            "Epoch 30/40\n",
            "4/4 [==============================] - 0s 84ms/step - loss: 0.0102 - accuracy: 1.0000\n",
            "Epoch 31/40\n",
            "4/4 [==============================] - 0s 86ms/step - loss: 0.0080 - accuracy: 1.0000\n",
            "Epoch 32/40\n",
            "4/4 [==============================] - 0s 81ms/step - loss: 0.0060 - accuracy: 1.0000\n",
            "Epoch 33/40\n",
            "4/4 [==============================] - 0s 82ms/step - loss: 0.0053 - accuracy: 1.0000\n",
            "Epoch 34/40\n",
            "4/4 [==============================] - 0s 89ms/step - loss: 0.0045 - accuracy: 1.0000\n",
            "Epoch 35/40\n",
            "4/4 [==============================] - 0s 82ms/step - loss: 0.0038 - accuracy: 1.0000\n",
            "Epoch 36/40\n",
            "4/4 [==============================] - 0s 84ms/step - loss: 0.0034 - accuracy: 1.0000\n",
            "Epoch 37/40\n",
            "4/4 [==============================] - 0s 86ms/step - loss: 0.0030 - accuracy: 1.0000\n",
            "Epoch 38/40\n",
            "4/4 [==============================] - 0s 88ms/step - loss: 0.0027 - accuracy: 1.0000\n",
            "Epoch 39/40\n",
            "4/4 [==============================] - 0s 81ms/step - loss: 0.0024 - accuracy: 1.0000\n",
            "Epoch 40/40\n",
            "4/4 [==============================] - 0s 90ms/step - loss: 0.0022 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7a23cab37190>"
            ]
          },
          "metadata": {},
          "execution_count": 52
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
        "outputId": "29245a90-2a69-42a5-87f4-c0a451a73fff"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2/2 [==============================] - 0s 14ms/step\n"
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
        "outputId": "29bc91cb-2981-4d30-cc51-6594f49570cc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9180327868852459, 0.912280701754386, 0.8357565966612817, 0.8398445462992558)"
            ]
          },
          "metadata": {},
          "execution_count": 54
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
        "outputId": "d93cd9e5-b414-4f0f-cc05-05b959bfcd41"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8823529411764706, 0.9629629629629629)"
            ]
          },
          "metadata": {},
          "execution_count": 55
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
        "outputId": "4a75fd0a-b84e-4d33-9ebc-7b414da1aea7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "5/5 [==============================] - 2s 72ms/step - loss: 0.6933 - accuracy: 0.4838\n",
            "Epoch 2/40\n",
            "5/5 [==============================] - 0s 72ms/step - loss: 0.6818 - accuracy: 0.6104\n",
            "Epoch 3/40\n",
            "5/5 [==============================] - 0s 70ms/step - loss: 0.6503 - accuracy: 0.6688\n",
            "Epoch 4/40\n",
            "5/5 [==============================] - 0s 71ms/step - loss: 0.5957 - accuracy: 0.7240\n",
            "Epoch 5/40\n",
            "5/5 [==============================] - 0s 73ms/step - loss: 0.5574 - accuracy: 0.7435\n",
            "Epoch 6/40\n",
            "5/5 [==============================] - 0s 72ms/step - loss: 0.5378 - accuracy: 0.7013\n",
            "Epoch 7/40\n",
            "5/5 [==============================] - 0s 70ms/step - loss: 0.5016 - accuracy: 0.7695\n",
            "Epoch 8/40\n",
            "5/5 [==============================] - 0s 74ms/step - loss: 0.4843 - accuracy: 0.7825\n",
            "Epoch 9/40\n",
            "5/5 [==============================] - 0s 72ms/step - loss: 0.4663 - accuracy: 0.7890\n",
            "Epoch 10/40\n",
            "5/5 [==============================] - 0s 72ms/step - loss: 0.4463 - accuracy: 0.7987\n",
            "Epoch 11/40\n",
            "5/5 [==============================] - 0s 76ms/step - loss: 0.4362 - accuracy: 0.7987\n",
            "Epoch 12/40\n",
            "5/5 [==============================] - 0s 69ms/step - loss: 0.4224 - accuracy: 0.8182\n",
            "Epoch 13/40\n",
            "5/5 [==============================] - 0s 71ms/step - loss: 0.4277 - accuracy: 0.8117\n",
            "Epoch 14/40\n",
            "5/5 [==============================] - 0s 76ms/step - loss: 0.3878 - accuracy: 0.8539\n",
            "Epoch 15/40\n",
            "5/5 [==============================] - 0s 71ms/step - loss: 0.3860 - accuracy: 0.8149\n",
            "Epoch 16/40\n",
            "5/5 [==============================] - 0s 71ms/step - loss: 0.3462 - accuracy: 0.8636\n",
            "Epoch 17/40\n",
            "5/5 [==============================] - 0s 81ms/step - loss: 0.3135 - accuracy: 0.8701\n",
            "Epoch 18/40\n",
            "5/5 [==============================] - 0s 69ms/step - loss: 0.3162 - accuracy: 0.8734\n",
            "Epoch 19/40\n",
            "5/5 [==============================] - 0s 69ms/step - loss: 0.2609 - accuracy: 0.8994\n",
            "Epoch 20/40\n",
            "5/5 [==============================] - 0s 71ms/step - loss: 0.2597 - accuracy: 0.8831\n",
            "Epoch 21/40\n",
            "5/5 [==============================] - 0s 71ms/step - loss: 0.2771 - accuracy: 0.8766\n",
            "Epoch 22/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.2079 - accuracy: 0.9383\n",
            "Epoch 23/40\n",
            "5/5 [==============================] - 1s 115ms/step - loss: 0.1783 - accuracy: 0.9253\n",
            "Epoch 24/40\n",
            "5/5 [==============================] - 1s 116ms/step - loss: 0.1639 - accuracy: 0.9286\n",
            "Epoch 25/40\n",
            "5/5 [==============================] - 1s 119ms/step - loss: 0.1332 - accuracy: 0.9675\n",
            "Epoch 26/40\n",
            "5/5 [==============================] - 1s 118ms/step - loss: 0.1299 - accuracy: 0.9578\n",
            "Epoch 27/40\n",
            "5/5 [==============================] - 1s 119ms/step - loss: 0.0873 - accuracy: 0.9740\n",
            "Epoch 28/40\n",
            "5/5 [==============================] - 1s 118ms/step - loss: 0.0705 - accuracy: 0.9870\n",
            "Epoch 29/40\n",
            "5/5 [==============================] - 0s 74ms/step - loss: 0.0599 - accuracy: 0.9935\n",
            "Epoch 30/40\n",
            "5/5 [==============================] - 0s 72ms/step - loss: 0.0470 - accuracy: 0.9935\n",
            "Epoch 31/40\n",
            "5/5 [==============================] - 0s 70ms/step - loss: 0.0344 - accuracy: 0.9968\n",
            "Epoch 32/40\n",
            "5/5 [==============================] - 0s 70ms/step - loss: 0.0335 - accuracy: 0.9968\n",
            "Epoch 33/40\n",
            "5/5 [==============================] - 0s 73ms/step - loss: 0.0231 - accuracy: 0.9968\n",
            "Epoch 34/40\n",
            "5/5 [==============================] - 0s 70ms/step - loss: 0.0206 - accuracy: 1.0000\n",
            "Epoch 35/40\n",
            "5/5 [==============================] - 0s 74ms/step - loss: 0.0182 - accuracy: 1.0000\n",
            "Epoch 36/40\n",
            "5/5 [==============================] - 0s 73ms/step - loss: 0.0103 - accuracy: 1.0000\n",
            "Epoch 37/40\n",
            "5/5 [==============================] - 0s 71ms/step - loss: 0.0079 - accuracy: 1.0000\n",
            "Epoch 38/40\n",
            "5/5 [==============================] - 0s 71ms/step - loss: 0.0066 - accuracy: 1.0000\n",
            "Epoch 39/40\n",
            "5/5 [==============================] - 0s 73ms/step - loss: 0.0057 - accuracy: 1.0000\n",
            "Epoch 40/40\n",
            "5/5 [==============================] - 0s 69ms/step - loss: 0.0046 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7a23caf90d60>"
            ]
          },
          "metadata": {},
          "execution_count": 64
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
        "outputId": "f0b94aa8-9693-492a-ff4f-599e612a9acb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5/5 [==============================] - 0s 15ms/step\n"
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
        "outputId": "40cc293b-3b90-49cd-874e-73e86e8c81c3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9318181818181818,\n",
              " 0.9291338582677167,\n",
              " 0.8635110294117647,\n",
              " 0.8644061166303924)"
            ]
          },
          "metadata": {},
          "execution_count": 66
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
        "outputId": "09ca5434-ec2d-430f-a986-9158a54097e4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9552238805970149, 0.9076923076923077)"
            ]
          },
          "metadata": {},
          "execution_count": 67
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [
        "2fpMtcJ5mR7-"
      ],
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

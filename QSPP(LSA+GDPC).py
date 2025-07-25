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
        "df = pd.read_csv('/content/LSA+GDPC.csv')\n",
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
        "outputId": "f00a63de-5c32-44ed-c97f-d07e45a19c7f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                                          Classifier  Accuracy       mcc  \\\n",
            "0  RandomForestClassifier(max_depth=5, n_estimato...  0.837662  0.675484   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.896104  0.792329   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.909091  0.818113   \n",
            "3  GradientBoostingClassifier(learning_rate=0.5, ...  0.889610  0.779302   \n",
            "4  AdaBoostClassifier(learning_rate=0.1, n_estima...  0.847403  0.695097   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.675256   0.823529  0.845638  0.834437     0.830189     0.845638  \n",
            "1  0.791725   0.909091  0.872483  0.890411     0.918239     0.872483  \n",
            "2  0.817836   0.917241  0.892617  0.904762     0.924528     0.892617  \n",
            "3  0.778708   0.902098  0.865772  0.883562     0.911950     0.865772  \n",
            "4  0.693902   0.864286  0.812081  0.837370     0.880503     0.812081  \n"
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
        "total_Metics.to_csv(\"total_Metics((LSA+GDPC)-CV).csv\")\n",
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
        "outputId": "cc0ac60b-f097-4bf4-aefa-cc2e70194bc9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Info] Number of positive: 158, number of negative: 150\n",
            "[LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000814 seconds.\n",
            "You can set `force_col_wise=true` to remove the overhead.\n",
            "[LightGBM] [Info] Total Bins 13973\n",
            "[LightGBM] [Info] Number of data points in the train set: 308, number of used features: 153\n",
            "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.512987 -> initscore=0.051960\n",
            "[LightGBM] [Info] Start training from score 0.051960\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
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
            "0  (DecisionTreeClassifier(max_depth=5, max_featu...  0.825758  0.659209   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.901515  0.802579   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.946970  0.893750   \n",
            "3  ([DecisionTreeRegressor(criterion='friedman_ms...  0.924242  0.848039   \n",
            "4  (DecisionTreeClassifier(max_depth=1, random_st...  0.909091  0.818224   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.653108   0.774648  0.887097  0.827068     0.771429     0.887097  \n",
            "1  0.802486   0.888889  0.903226  0.896000     0.900000     0.903226  \n",
            "2  0.893646   0.936508  0.951613  0.944000     0.942857     0.951613  \n",
            "3  0.847645   0.933333  0.903226  0.918033     0.942857     0.903226  \n",
            "4  0.817847   0.890625  0.919355  0.904762     0.900000     0.919355  \n"
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
        "total_Metics.to_csv(\"total_Metics((LSA+GDPC)-TS)).csv\")\n",
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
        "df = pd.read_csv('/content/Res-LSA+GDPC.csv')\n",
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
        "outputId": "d1bfa908-7311-4538-fc43-05dc859aaec0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "4/4 [==============================] - 2s 101ms/step - loss: 0.6938 - accuracy: 0.4413\n",
            "Epoch 2/40\n",
            "4/4 [==============================] - 0s 101ms/step - loss: 0.6869 - accuracy: 0.8421\n",
            "Epoch 3/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 0.6686 - accuracy: 0.7814\n",
            "Epoch 4/40\n",
            "4/4 [==============================] - 0s 98ms/step - loss: 0.6252 - accuracy: 0.8583\n",
            "Epoch 5/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 0.5372 - accuracy: 0.8664\n",
            "Epoch 6/40\n",
            "4/4 [==============================] - 0s 103ms/step - loss: 0.4242 - accuracy: 0.8502\n",
            "Epoch 7/40\n",
            "4/4 [==============================] - 0s 91ms/step - loss: 0.3469 - accuracy: 0.8785\n",
            "Epoch 8/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 0.2822 - accuracy: 0.8988\n",
            "Epoch 9/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 0.2816 - accuracy: 0.8826\n",
            "Epoch 10/40\n",
            "4/4 [==============================] - 0s 91ms/step - loss: 0.2048 - accuracy: 0.9352\n",
            "Epoch 11/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 0.1622 - accuracy: 0.9312\n",
            "Epoch 12/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 0.1308 - accuracy: 0.9514\n",
            "Epoch 13/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 0.1312 - accuracy: 0.9352\n",
            "Epoch 14/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 0.0858 - accuracy: 0.9555\n",
            "Epoch 15/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 0.0469 - accuracy: 0.9879\n",
            "Epoch 16/40\n",
            "4/4 [==============================] - 0s 95ms/step - loss: 0.0254 - accuracy: 1.0000\n",
            "Epoch 17/40\n",
            "4/4 [==============================] - 0s 101ms/step - loss: 0.0164 - accuracy: 1.0000\n",
            "Epoch 18/40\n",
            "4/4 [==============================] - 0s 99ms/step - loss: 0.0118 - accuracy: 1.0000\n",
            "Epoch 19/40\n",
            "4/4 [==============================] - 0s 99ms/step - loss: 0.0077 - accuracy: 1.0000\n",
            "Epoch 20/40\n",
            "4/4 [==============================] - 0s 127ms/step - loss: 0.0041 - accuracy: 1.0000\n",
            "Epoch 21/40\n",
            "4/4 [==============================] - 1s 171ms/step - loss: 0.0034 - accuracy: 1.0000\n",
            "Epoch 22/40\n",
            "4/4 [==============================] - 1s 168ms/step - loss: 0.0020 - accuracy: 1.0000\n",
            "Epoch 23/40\n",
            "4/4 [==============================] - 1s 163ms/step - loss: 0.0018 - accuracy: 1.0000\n",
            "Epoch 24/40\n",
            "4/4 [==============================] - 1s 165ms/step - loss: 0.0013 - accuracy: 1.0000\n",
            "Epoch 25/40\n",
            "4/4 [==============================] - 0s 102ms/step - loss: 0.0011 - accuracy: 1.0000\n",
            "Epoch 26/40\n",
            "4/4 [==============================] - 0s 95ms/step - loss: 9.2039e-04 - accuracy: 1.0000\n",
            "Epoch 27/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 8.0091e-04 - accuracy: 1.0000\n",
            "Epoch 28/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 7.3349e-04 - accuracy: 1.0000\n",
            "Epoch 29/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 6.4079e-04 - accuracy: 1.0000\n",
            "Epoch 30/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 5.9078e-04 - accuracy: 1.0000\n",
            "Epoch 31/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 5.5289e-04 - accuracy: 1.0000\n",
            "Epoch 32/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 5.0816e-04 - accuracy: 1.0000\n",
            "Epoch 33/40\n",
            "4/4 [==============================] - 0s 91ms/step - loss: 4.7371e-04 - accuracy: 1.0000\n",
            "Epoch 34/40\n",
            "4/4 [==============================] - 0s 91ms/step - loss: 4.4389e-04 - accuracy: 1.0000\n",
            "Epoch 35/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 4.1681e-04 - accuracy: 1.0000\n",
            "Epoch 36/40\n",
            "4/4 [==============================] - 0s 93ms/step - loss: 3.9487e-04 - accuracy: 1.0000\n",
            "Epoch 37/40\n",
            "4/4 [==============================] - 0s 96ms/step - loss: 3.7110e-04 - accuracy: 1.0000\n",
            "Epoch 38/40\n",
            "4/4 [==============================] - 0s 96ms/step - loss: 3.5462e-04 - accuracy: 1.0000\n",
            "Epoch 39/40\n",
            "4/4 [==============================] - 0s 90ms/step - loss: 3.3388e-04 - accuracy: 1.0000\n",
            "Epoch 40/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 3.1682e-04 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7afe743337c0>"
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
        "outputId": "f588b7a4-cf13-49aa-f143-fb3ffd8c71a5"
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
        "outputId": "493aa778-6e25-4ec0-89c5-031719841f90"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9672131147540983, 0.967741935483871, 0.9344790547798066, 0.9364913929301529)"
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
        "outputId": "ec602550-a215-439b-dbda-221b0ed1e63d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9354838709677419, 1.0)"
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
        "outputId": "36f26e2e-a090-48da-ae10-80449fa8aa3b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "5/5 [==============================] - 2s 85ms/step - loss: 0.6922 - accuracy: 0.5422\n",
            "Epoch 2/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 0.6762 - accuracy: 0.5552\n",
            "Epoch 3/40\n",
            "5/5 [==============================] - 0s 87ms/step - loss: 0.6501 - accuracy: 0.5649\n",
            "Epoch 4/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 0.6058 - accuracy: 0.7792\n",
            "Epoch 5/40\n",
            "5/5 [==============================] - 0s 87ms/step - loss: 0.5178 - accuracy: 0.8052\n",
            "Epoch 6/40\n",
            "5/5 [==============================] - 0s 81ms/step - loss: 0.4228 - accuracy: 0.8474\n",
            "Epoch 7/40\n",
            "5/5 [==============================] - 0s 92ms/step - loss: 0.3569 - accuracy: 0.8377\n",
            "Epoch 8/40\n",
            "5/5 [==============================] - 0s 84ms/step - loss: 0.2684 - accuracy: 0.8994\n",
            "Epoch 9/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 0.1850 - accuracy: 0.9253\n",
            "Epoch 10/40\n",
            "5/5 [==============================] - 0s 90ms/step - loss: 0.1413 - accuracy: 0.9481\n",
            "Epoch 11/40\n",
            "5/5 [==============================] - 0s 87ms/step - loss: 0.1008 - accuracy: 0.9740\n",
            "Epoch 12/40\n",
            "5/5 [==============================] - 0s 88ms/step - loss: 0.0708 - accuracy: 0.9838\n",
            "Epoch 13/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 0.0429 - accuracy: 0.9903\n",
            "Epoch 14/40\n",
            "5/5 [==============================] - 1s 143ms/step - loss: 0.0295 - accuracy: 0.9935\n",
            "Epoch 15/40\n",
            "5/5 [==============================] - 1s 146ms/step - loss: 0.0180 - accuracy: 1.0000\n",
            "Epoch 16/40\n",
            "5/5 [==============================] - 1s 146ms/step - loss: 0.0150 - accuracy: 1.0000\n",
            "Epoch 17/40\n",
            "5/5 [==============================] - 1s 153ms/step - loss: 0.0181 - accuracy: 0.9968\n",
            "Epoch 18/40\n",
            "5/5 [==============================] - 1s 113ms/step - loss: 0.0082 - accuracy: 1.0000\n",
            "Epoch 19/40\n",
            "5/5 [==============================] - 0s 88ms/step - loss: 0.0054 - accuracy: 1.0000\n",
            "Epoch 20/40\n",
            "5/5 [==============================] - 0s 86ms/step - loss: 0.0065 - accuracy: 1.0000\n",
            "Epoch 21/40\n",
            "5/5 [==============================] - 0s 86ms/step - loss: 0.0052 - accuracy: 1.0000\n",
            "Epoch 22/40\n",
            "5/5 [==============================] - 0s 84ms/step - loss: 0.0029 - accuracy: 1.0000\n",
            "Epoch 23/40\n",
            "5/5 [==============================] - 0s 88ms/step - loss: 0.0018 - accuracy: 1.0000\n",
            "Epoch 24/40\n",
            "5/5 [==============================] - 0s 84ms/step - loss: 0.0016 - accuracy: 1.0000\n",
            "Epoch 25/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.0011 - accuracy: 1.0000\n",
            "Epoch 26/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.0011 - accuracy: 1.0000\n",
            "Epoch 27/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 9.2524e-04 - accuracy: 1.0000\n",
            "Epoch 28/40\n",
            "5/5 [==============================] - 0s 86ms/step - loss: 8.3202e-04 - accuracy: 1.0000\n",
            "Epoch 29/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 7.2788e-04 - accuracy: 1.0000\n",
            "Epoch 30/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 6.6349e-04 - accuracy: 1.0000\n",
            "Epoch 31/40\n",
            "5/5 [==============================] - 0s 91ms/step - loss: 6.1296e-04 - accuracy: 1.0000\n",
            "Epoch 32/40\n",
            "5/5 [==============================] - 0s 86ms/step - loss: 5.7444e-04 - accuracy: 1.0000\n",
            "Epoch 33/40\n",
            "5/5 [==============================] - 0s 90ms/step - loss: 5.3366e-04 - accuracy: 1.0000\n",
            "Epoch 34/40\n",
            "5/5 [==============================] - 0s 87ms/step - loss: 5.0938e-04 - accuracy: 1.0000\n",
            "Epoch 35/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 4.8975e-04 - accuracy: 1.0000\n",
            "Epoch 36/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 4.6206e-04 - accuracy: 1.0000\n",
            "Epoch 37/40\n",
            "5/5 [==============================] - 0s 87ms/step - loss: 4.3437e-04 - accuracy: 1.0000\n",
            "Epoch 38/40\n",
            "5/5 [==============================] - 0s 88ms/step - loss: 4.0960e-04 - accuracy: 1.0000\n",
            "Epoch 39/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 3.9172e-04 - accuracy: 1.0000\n",
            "Epoch 40/40\n",
            "5/5 [==============================] - 0s 92ms/step - loss: 3.7802e-04 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7afe61fc5810>"
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
        "outputId": "e56ca6c5-0f48-472d-f04b-ae45c3322a46"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5/5 [==============================] - 0s 11ms/step\n"
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
        "outputId": "55716565-aa44-433f-fab3-ad8e217ea64c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9621212121212122,\n",
              " 0.9635036496350364,\n",
              " 0.9241727941176471,\n",
              " 0.9251307613324742)"
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
        "outputId": "c3b43dd6-c7e7-44a4-db01-7d86b91ae91c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9384615384615385, 0.9850746268656716)"
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

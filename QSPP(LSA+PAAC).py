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
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "B_9sc_sxlvZ-",
        "outputId": "cc1e56b3-2534-4dd2-aff7-23c98ce54287"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting catboost\n",
            "  Downloading catboost-1.2.1-cp310-cp310-manylinux2014_x86_64.whl (98.7 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m98.7/98.7 MB\u001b[0m \u001b[31m9.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: graphviz in /usr/local/lib/python3.10/dist-packages (from catboost) (0.20.1)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from catboost) (3.7.1)\n",
            "Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.10/dist-packages (from catboost) (1.23.5)\n",
            "Requirement already satisfied: pandas>=0.24 in /usr/local/lib/python3.10/dist-packages (from catboost) (1.5.3)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from catboost) (1.11.2)\n",
            "Requirement already satisfied: plotly in /usr/local/lib/python3.10/dist-packages (from catboost) (5.15.0)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from catboost) (1.16.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.24->catboost) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.24->catboost) (2023.3.post1)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (1.1.0)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (0.11.0)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (4.42.1)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (1.4.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (23.1)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (9.4.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->catboost) (3.1.1)\n",
            "Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from plotly->catboost) (8.2.3)\n",
            "Installing collected packages: catboost\n",
            "Successfully installed catboost-1.2.1\n"
          ]
        }
      ],
      "source": [
        "!pip install catboost"
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
        "from catboost import CatBoostClassifier\n",
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
        "df = pd.read_csv('/content/LSA+PAAC.csv')\n",
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
        "outputId": "34dd40f9-b205-4c2e-ea35-b3f015c27a45"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                                          Classifier  Accuracy       mcc  \\\n",
            "0  RandomForestClassifier(max_depth=5, n_estimato...  0.785714  0.574806   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.808442  0.616896   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.808442  0.616896   \n",
            "3  GradientBoostingClassifier(learning_rate=0.5, ...  0.834416  0.668817   \n",
            "4  AdaBoostClassifier(learning_rate=0.1, n_estima...  0.827922  0.659178   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.571699   0.757396  0.836601  0.795031     0.735484     0.836601  \n",
            "1  0.616883   0.805195  0.810458  0.807818     0.806452     0.810458  \n",
            "2  0.616883   0.805195  0.810458  0.807818     0.806452     0.810458  \n",
            "3  0.668803   0.835526  0.830065  0.832787     0.838710     0.830065  \n",
            "4  0.656047   0.797619  0.875817  0.834891     0.780645     0.875817  \n"
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
        "          AdaBoostClassifier(n_estimators = 200, learning_rate = 0.1, random_state = 50),\n",
        "          Stacking]\n",
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
        "total_Metics.to_csv(\"total_Metics(LSA-CV).csv\")\n",
        "clf = StackingClassifier( estimators=estimator, final_estimator=RandomForestClassifier(n_estimators = 200, max_depth = 5))\n",
        "prob = clf.fit_transform(xtrain, ytrain)\n",
        "pd.DataFrame(prob).to_csv(\"total_Metics_Probability.csv\")"
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
        "outputId": "a5e900c4-4e7f-494e-95ec-7e286cbfd5f4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Info] Number of positive: 153, number of negative: 155\n",
            "[LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000645 seconds.\n",
            "You can set `force_col_wise=true` to remove the overhead.\n",
            "[LightGBM] [Info] Total Bins 12249\n",
            "[LightGBM] [Info] Number of data points in the train set: 308, number of used features: 128\n",
            "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.496753 -> initscore=-0.012987\n",
            "[LightGBM] [Info] Start training from score -0.012987\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
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
            "0  (DecisionTreeClassifier(max_depth=5, max_featu...  0.848485  0.699387   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.856061  0.712990   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.871212  0.742509   \n",
            "3  ([DecisionTreeRegressor(criterion='friedman_ms...  0.901515  0.803032   \n",
            "4  (DecisionTreeClassifier(max_depth=1, random_st...  0.909091  0.818433   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.696482   0.821918  0.895522  0.857143     0.800000     0.895522  \n",
            "1  0.712253   0.875000  0.835821  0.854962     0.876923     0.835821  \n",
            "2  0.742424   0.878788  0.865672  0.872180     0.876923     0.865672  \n",
            "3  0.802940   0.897059  0.910448  0.903704     0.892308     0.910448  \n",
            "4  0.818057   0.898551  0.925373  0.911765     0.892308     0.925373  \n"
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
        "total_Metics.to_csv(\"total_Metics((LSA+PAAC)-TS)).csv\")\n",
        "# clf = StackingClassifier( estimators=estimator, final_estimator=RandomForestClassifier(n_estimators = 200, max_depth = 5))\n",
        "# prob = clf.fit_transform(xtest, ytest)\n",
        "# pd.DataFrame(prob).to_csv(\"total_Metics_Probability(LSA+PAAC).csv\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "models = [RandomForestClassifier(n_estimators = 200, max_depth = 5),\n",
        "          CatBoostClassifier(depth= 5, iterations = 35, learning_rate = 0.35),\n",
        "          XGBClassifier(n_estimators = 200,max_depth = 5, learning_rate = 0.1),\n",
        "          LGBMClassifier(learning_rate = 0.1,max_depth = 5,random_state = 50),\n",
        "          GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.5, random_state = 50),\n",
        "          AdaBoostClassifier(n_estimators = 200, learning_rate = 0.1, random_state = 50)]"
      ],
      "metadata": {
        "id": "ySkxVOmYLxqy"
      },
      "execution_count": null,
      "outputs": []
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
        "df = pd.read_csv('/content/Res-LSA+PAAC.csv')\n",
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
        "outputId": "61d0e886-f741-4967-ac3f-95e19409182c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "4/4 [==============================] - 2s 57ms/step - loss: 0.6870 - accuracy: 0.5506\n",
            "Epoch 2/40\n",
            "4/4 [==============================] - 0s 52ms/step - loss: 0.6765 - accuracy: 0.5506\n",
            "Epoch 3/40\n",
            "4/4 [==============================] - 0s 52ms/step - loss: 0.6580 - accuracy: 0.5506\n",
            "Epoch 4/40\n",
            "4/4 [==============================] - 0s 53ms/step - loss: 0.6352 - accuracy: 0.5506\n",
            "Epoch 5/40\n",
            "4/4 [==============================] - 0s 60ms/step - loss: 0.6050 - accuracy: 0.7247\n",
            "Epoch 6/40\n",
            "4/4 [==============================] - 0s 73ms/step - loss: 0.5795 - accuracy: 0.8138\n",
            "Epoch 7/40\n",
            "4/4 [==============================] - 0s 80ms/step - loss: 0.5393 - accuracy: 0.8502\n",
            "Epoch 8/40\n",
            "4/4 [==============================] - 0s 79ms/step - loss: 0.4872 - accuracy: 0.8583\n",
            "Epoch 9/40\n",
            "4/4 [==============================] - 0s 72ms/step - loss: 0.4423 - accuracy: 0.8745\n",
            "Epoch 10/40\n",
            "4/4 [==============================] - 0s 80ms/step - loss: 0.3936 - accuracy: 0.8907\n",
            "Epoch 11/40\n",
            "4/4 [==============================] - 0s 73ms/step - loss: 0.3474 - accuracy: 0.9069\n",
            "Epoch 12/40\n",
            "4/4 [==============================] - 0s 55ms/step - loss: 0.3073 - accuracy: 0.9190\n",
            "Epoch 13/40\n",
            "4/4 [==============================] - 0s 56ms/step - loss: 0.3109 - accuracy: 0.9028\n",
            "Epoch 14/40\n",
            "4/4 [==============================] - 0s 48ms/step - loss: 0.2946 - accuracy: 0.9028\n",
            "Epoch 15/40\n",
            "4/4 [==============================] - 0s 49ms/step - loss: 0.2657 - accuracy: 0.9190\n",
            "Epoch 16/40\n",
            "4/4 [==============================] - 0s 48ms/step - loss: 0.2481 - accuracy: 0.9352\n",
            "Epoch 17/40\n",
            "4/4 [==============================] - 0s 48ms/step - loss: 0.2451 - accuracy: 0.9271\n",
            "Epoch 18/40\n",
            "4/4 [==============================] - 0s 52ms/step - loss: 0.2435 - accuracy: 0.9312\n",
            "Epoch 19/40\n",
            "4/4 [==============================] - 0s 46ms/step - loss: 0.2326 - accuracy: 0.9352\n",
            "Epoch 20/40\n",
            "4/4 [==============================] - 0s 48ms/step - loss: 0.2276 - accuracy: 0.9433\n",
            "Epoch 21/40\n",
            "4/4 [==============================] - 0s 48ms/step - loss: 0.2350 - accuracy: 0.9312\n",
            "Epoch 22/40\n",
            "4/4 [==============================] - 0s 50ms/step - loss: 0.2281 - accuracy: 0.9393\n",
            "Epoch 23/40\n",
            "4/4 [==============================] - 0s 50ms/step - loss: 0.2104 - accuracy: 0.9474\n",
            "Epoch 24/40\n",
            "4/4 [==============================] - 0s 46ms/step - loss: 0.2042 - accuracy: 0.9474\n",
            "Epoch 25/40\n",
            "4/4 [==============================] - 0s 48ms/step - loss: 0.1997 - accuracy: 0.9514\n",
            "Epoch 26/40\n",
            "4/4 [==============================] - 0s 50ms/step - loss: 0.1982 - accuracy: 0.9514\n",
            "Epoch 27/40\n",
            "4/4 [==============================] - 0s 46ms/step - loss: 0.1964 - accuracy: 0.9514\n",
            "Epoch 28/40\n",
            "4/4 [==============================] - 0s 47ms/step - loss: 0.1969 - accuracy: 0.9514\n",
            "Epoch 29/40\n",
            "4/4 [==============================] - 0s 47ms/step - loss: 0.1959 - accuracy: 0.9514\n",
            "Epoch 30/40\n",
            "4/4 [==============================] - 0s 50ms/step - loss: 0.1956 - accuracy: 0.9514\n",
            "Epoch 31/40\n",
            "4/4 [==============================] - 0s 53ms/step - loss: 0.1954 - accuracy: 0.9514\n",
            "Epoch 32/40\n",
            "4/4 [==============================] - 0s 47ms/step - loss: 0.1952 - accuracy: 0.9514\n",
            "Epoch 33/40\n",
            "4/4 [==============================] - 0s 50ms/step - loss: 0.1952 - accuracy: 0.9514\n",
            "Epoch 34/40\n",
            "4/4 [==============================] - 0s 50ms/step - loss: 0.1950 - accuracy: 0.9514\n",
            "Epoch 35/40\n",
            "4/4 [==============================] - 0s 50ms/step - loss: 0.1950 - accuracy: 0.9514\n",
            "Epoch 36/40\n",
            "4/4 [==============================] - 0s 52ms/step - loss: 0.1951 - accuracy: 0.9514\n",
            "Epoch 37/40\n",
            "4/4 [==============================] - 0s 54ms/step - loss: 0.1948 - accuracy: 0.9514\n",
            "Epoch 38/40\n",
            "4/4 [==============================] - 0s 50ms/step - loss: 0.1949 - accuracy: 0.9514\n",
            "Epoch 39/40\n",
            "4/4 [==============================] - 0s 53ms/step - loss: 0.1948 - accuracy: 0.9514\n",
            "Epoch 40/40\n",
            "4/4 [==============================] - 0s 57ms/step - loss: 0.1949 - accuracy: 0.9514\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7d377c305ae0>"
            ]
          },
          "metadata": {},
          "execution_count": 44
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
        "outputId": "acdc9740-6c2e-484d-a8d3-39f3d5b1c189"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2/2 [==============================] - 0s 15ms/step\n"
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
        "outputId": "68ce147a-4532-4ede-c295-b901f45042a9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9672131147540983,\n",
              " 0.9666666666666667,\n",
              " 0.9344086021505377,\n",
              " 0.9344086021505377)"
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
        "outputId": "2ef6c568-ea55-43ac-abb8-45d990ebf8fb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.967741935483871, 0.9666666666666667)"
            ]
          },
          "metadata": {},
          "execution_count": 47
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
        "outputId": "b764d912-5156-4430-d021-0d35cf4d37f9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "5/5 [==============================] - 1s 45ms/step - loss: 0.6601 - accuracy: 0.5487\n",
            "Epoch 2/40\n",
            "5/5 [==============================] - 0s 43ms/step - loss: 0.5792 - accuracy: 0.5617\n",
            "Epoch 3/40\n",
            "5/5 [==============================] - 0s 46ms/step - loss: 0.5053 - accuracy: 0.7630\n",
            "Epoch 4/40\n",
            "5/5 [==============================] - 0s 45ms/step - loss: 0.4097 - accuracy: 0.8247\n",
            "Epoch 5/40\n",
            "5/5 [==============================] - 0s 47ms/step - loss: 0.4038 - accuracy: 0.8117\n",
            "Epoch 6/40\n",
            "5/5 [==============================] - 0s 44ms/step - loss: 0.4168 - accuracy: 0.8214\n",
            "Epoch 7/40\n",
            "5/5 [==============================] - 0s 44ms/step - loss: 0.4117 - accuracy: 0.8052\n",
            "Epoch 8/40\n",
            "5/5 [==============================] - 0s 46ms/step - loss: 0.3639 - accuracy: 0.8604\n",
            "Epoch 9/40\n",
            "5/5 [==============================] - 0s 45ms/step - loss: 0.3475 - accuracy: 0.8604\n",
            "Epoch 10/40\n",
            "5/5 [==============================] - 0s 42ms/step - loss: 0.3290 - accuracy: 0.8799\n",
            "Epoch 11/40\n",
            "5/5 [==============================] - 0s 46ms/step - loss: 0.3079 - accuracy: 0.8831\n",
            "Epoch 12/40\n",
            "5/5 [==============================] - 0s 44ms/step - loss: 0.2886 - accuracy: 0.8929\n",
            "Epoch 13/40\n",
            "5/5 [==============================] - 0s 45ms/step - loss: 0.2663 - accuracy: 0.8994\n",
            "Epoch 14/40\n",
            "5/5 [==============================] - 0s 43ms/step - loss: 0.2585 - accuracy: 0.9123\n",
            "Epoch 15/40\n",
            "5/5 [==============================] - 0s 46ms/step - loss: 0.2357 - accuracy: 0.9058\n",
            "Epoch 16/40\n",
            "5/5 [==============================] - 0s 45ms/step - loss: 0.2221 - accuracy: 0.9188\n",
            "Epoch 17/40\n",
            "5/5 [==============================] - 0s 45ms/step - loss: 0.1889 - accuracy: 0.9351\n",
            "Epoch 18/40\n",
            "5/5 [==============================] - 0s 43ms/step - loss: 0.1652 - accuracy: 0.9318\n",
            "Epoch 19/40\n",
            "5/5 [==============================] - 0s 45ms/step - loss: 0.1458 - accuracy: 0.9383\n",
            "Epoch 20/40\n",
            "5/5 [==============================] - 0s 43ms/step - loss: 0.1225 - accuracy: 0.9643\n",
            "Epoch 21/40\n",
            "5/5 [==============================] - 0s 46ms/step - loss: 0.1006 - accuracy: 0.9708\n",
            "Epoch 22/40\n",
            "5/5 [==============================] - 0s 46ms/step - loss: 0.0860 - accuracy: 0.9805\n",
            "Epoch 23/40\n",
            "5/5 [==============================] - 0s 43ms/step - loss: 0.0898 - accuracy: 0.9773\n",
            "Epoch 24/40\n",
            "5/5 [==============================] - 0s 46ms/step - loss: 0.1031 - accuracy: 0.9610\n",
            "Epoch 25/40\n",
            "5/5 [==============================] - 0s 45ms/step - loss: 0.0545 - accuracy: 0.9773\n",
            "Epoch 26/40\n",
            "5/5 [==============================] - 0s 49ms/step - loss: 0.0409 - accuracy: 0.9935\n",
            "Epoch 27/40\n",
            "5/5 [==============================] - 0s 67ms/step - loss: 0.0261 - accuracy: 0.9968\n",
            "Epoch 28/40\n",
            "5/5 [==============================] - 0s 65ms/step - loss: 0.0215 - accuracy: 1.0000\n",
            "Epoch 29/40\n",
            "5/5 [==============================] - 0s 68ms/step - loss: 0.0158 - accuracy: 1.0000\n",
            "Epoch 30/40\n",
            "5/5 [==============================] - 0s 66ms/step - loss: 0.0125 - accuracy: 1.0000\n",
            "Epoch 31/40\n",
            "5/5 [==============================] - 0s 68ms/step - loss: 0.0090 - accuracy: 1.0000\n",
            "Epoch 32/40\n",
            "5/5 [==============================] - 0s 65ms/step - loss: 0.0070 - accuracy: 1.0000\n",
            "Epoch 33/40\n",
            "5/5 [==============================] - 0s 46ms/step - loss: 0.0063 - accuracy: 1.0000\n",
            "Epoch 34/40\n",
            "5/5 [==============================] - 0s 45ms/step - loss: 0.0052 - accuracy: 1.0000\n",
            "Epoch 35/40\n",
            "5/5 [==============================] - 0s 42ms/step - loss: 0.0041 - accuracy: 1.0000\n",
            "Epoch 36/40\n",
            "5/5 [==============================] - 0s 42ms/step - loss: 0.0034 - accuracy: 1.0000\n",
            "Epoch 37/40\n",
            "5/5 [==============================] - 0s 44ms/step - loss: 0.0029 - accuracy: 1.0000\n",
            "Epoch 38/40\n",
            "5/5 [==============================] - 0s 44ms/step - loss: 0.0025 - accuracy: 1.0000\n",
            "Epoch 39/40\n",
            "5/5 [==============================] - 0s 45ms/step - loss: 0.0021 - accuracy: 1.0000\n",
            "Epoch 40/40\n",
            "5/5 [==============================] - 0s 44ms/step - loss: 0.0020 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7d377c0c3af0>"
            ]
          },
          "metadata": {},
          "execution_count": 56
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
        "outputId": "f2e88726-ae1f-4ecd-83d6-205faf233cc7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5/5 [==============================] - 0s 7ms/step\n"
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
        "outputId": "f0bcae77-fb43-4fc3-a3c4-82b47ab971b4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9696969696969697, 0.972972972972973, 0.9384902143522833, 0.9384902143522833)"
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
        "outputId": "35b02850-4b38-4132-9341-d31f4dd69f66"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9655172413793104, 0.972972972972973)"
            ]
          },
          "metadata": {},
          "execution_count": 59
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

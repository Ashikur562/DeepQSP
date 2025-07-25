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
      "execution_count": null,
      "metadata": {
        "id": "-Bz-7hYjwj-t"
      },
      "outputs": [],
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout,LSTM"
      ]
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
        "df = pd.read_csv('/content/FastText+GDPC.csv')\n",
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
      "execution_count": null,
      "metadata": {
        "id": "54qCUm3mGbDZ"
      },
      "outputs": [],
      "source": [
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "fUUuv7DhmBE0",
        "outputId": "10b545aa-c49f-4aa6-ff35-f91d8f367aca"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "                                          Classifier  Accuracy       mcc  \\\n",
            "0  RandomForestClassifier(max_depth=5, n_estimato...  0.833333  0.666595   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.843137  0.686221   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.826797  0.653520   \n",
            "3  GradientBoostingClassifier(learning_rate=0.5, ...  0.849673  0.699432   \n",
            "4  AdaBoostClassifier(learning_rate=0.1, n_estima...  0.820261  0.640492   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.666581   0.833333  0.827815  0.830565     0.838710     0.827815  \n",
            "1  0.686221   0.841060  0.841060  0.841060     0.845161     0.841060  \n",
            "2  0.653506   0.826667  0.821192  0.823920     0.832258     0.821192  \n",
            "3  0.699192   0.857143  0.834437  0.845638     0.864516     0.834437  \n",
            "4  0.640369   0.824324  0.807947  0.816054     0.832258     0.807947  \n"
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
        "total_Metics.to_csv(\"total_Metics((FT+GDPC)-CV).csv\")\n",
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
        "outputId": "176b42da-57a2-407d-bd48-8d0999cc2add"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).\n",
            "[LightGBM] [Info] Number of positive: 151, number of negative: 155\n",
            "[LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000876 seconds.\n",
            "You can set `force_col_wise=true` to remove the overhead.\n",
            "[LightGBM] [Info] Total Bins 13884\n",
            "[LightGBM] [Info] Number of data points in the train set: 306, number of used features: 153\n",
            "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.493464 -> initscore=-0.026145\n",
            "[LightGBM] [Info] Start training from score -0.026145\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
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
            "0  (DecisionTreeClassifier(max_depth=5, max_featu...  0.848485  0.696691   \n",
            "1  XGBClassifier(base_score=None, booster=None, c...  0.840909  0.684390   \n",
            "2       LGBMClassifier(max_depth=5, random_state=50)  0.878788  0.758308   \n",
            "3  ([DecisionTreeRegressor(criterion='friedman_ms...  0.886364  0.774234   \n",
            "4  (DecisionTreeClassifier(max_depth=1, random_st...  0.818182  0.637683   \n",
            "\n",
            "      Kappa  precision    recall        f1  sensitivity  specificity  \n",
            "0  0.696691   0.852941  0.852941  0.852941     0.843750     0.852941  \n",
            "1  0.680498   0.813333  0.897059  0.853147     0.781250     0.897059  \n",
            "2  0.756906   0.861111  0.911765  0.885714     0.843750     0.911765  \n",
            "3  0.771994   0.863014  0.926471  0.893617     0.843750     0.926471  \n",
            "4  0.635023   0.797297  0.867647  0.830986     0.765625     0.867647  \n"
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
        "total_Metics.to_csv(\"total_Metics((FT+GDPC)-TS)).csv\")\n",
        "# clf = StackingClassifier( estimators=estimator, final_estimator=RandomForestClassifier(n_estimators = 200, max_depth = 5))\n",
        "# prob = clf.fit_transform(xtest, ytest)\n",
        "# pd.DataFrame(prob).to_csv(\"total_Metics_Probability(LSA+PAAC).csv\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5zeJ0H9I373c"
      },
      "source": [
        "# **CNN**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mMLDGLBRAwfm"
      },
      "outputs": [],
      "source": [
        "df = pd.read_csv('/content/Res-FastText+GDPC.csv')\n",
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
      "cell_type": "markdown",
      "metadata": {
        "id": "3BLVQrcUB6rA"
      },
      "source": [
        "**Train**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Y3gZHPKDA0te"
      },
      "outputs": [],
      "source": [
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cv0C6ZF639bb"
      },
      "outputs": [],
      "source": [
        "xtrain = xtrain.to_numpy()\n",
        "ytrain = ytrain.to_numpy()\n",
        "xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)\n",
        "# xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MMwz6ZaP4NAN"
      },
      "outputs": [],
      "source": [
        "kf = KFold(n_splits=5, shuffle=True)\n",
        "for train_index, val_index in kf.split(xtrain):\n",
        "    X_train, X_val = xtrain[train_index], xtrain[val_index]\n",
        "    y_train, y_val = ytrain[train_index], ytrain[val_index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gAB9oIwJ4cie"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rGhc3hTl4fYT"
      },
      "outputs": [],
      "source": [
        "cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=2, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=2, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2gy_kUsg4mxD"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=2))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ldx-dq454sdt"
      },
      "outputs": [],
      "source": [
        "cnn.add(Flatten())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HhCvyAnq4uXb"
      },
      "outputs": [],
      "source": [
        "cnn.add(Dense(128, activation='relu'))\n",
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bPKlJAE04wWN"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hif7w0dG4yh0",
        "outputId": "b11d77a4-a459-467a-e7af-6ed1e3034d35"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "4/4 [==============================] - 2s 94ms/step - loss: 0.6925 - accuracy: 0.5347\n",
            "Epoch 2/40\n",
            "4/4 [==============================] - 0s 93ms/step - loss: 0.6927 - accuracy: 0.5224\n",
            "Epoch 3/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 0.6841 - accuracy: 0.5224\n",
            "Epoch 4/40\n",
            "4/4 [==============================] - 0s 93ms/step - loss: 0.6728 - accuracy: 0.5673\n",
            "Epoch 5/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 0.6484 - accuracy: 0.6367\n",
            "Epoch 6/40\n",
            "4/4 [==============================] - 0s 98ms/step - loss: 0.6133 - accuracy: 0.6857\n",
            "Epoch 7/40\n",
            "4/4 [==============================] - 0s 87ms/step - loss: 0.5559 - accuracy: 0.6939\n",
            "Epoch 8/40\n",
            "4/4 [==============================] - 0s 100ms/step - loss: 0.5207 - accuracy: 0.7265\n",
            "Epoch 9/40\n",
            "4/4 [==============================] - 1s 161ms/step - loss: 0.5147 - accuracy: 0.7224\n",
            "Epoch 10/40\n",
            "4/4 [==============================] - 1s 160ms/step - loss: 0.4997 - accuracy: 0.7224\n",
            "Epoch 11/40\n",
            "4/4 [==============================] - 1s 164ms/step - loss: 0.4700 - accuracy: 0.7959\n",
            "Epoch 12/40\n",
            "4/4 [==============================] - 1s 165ms/step - loss: 0.4350 - accuracy: 0.7878\n",
            "Epoch 13/40\n",
            "4/4 [==============================] - 1s 170ms/step - loss: 0.3975 - accuracy: 0.8286\n",
            "Epoch 14/40\n",
            "4/4 [==============================] - 0s 91ms/step - loss: 0.3958 - accuracy: 0.8327\n",
            "Epoch 15/40\n",
            "4/4 [==============================] - 0s 93ms/step - loss: 0.3589 - accuracy: 0.8449\n",
            "Epoch 16/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 0.3343 - accuracy: 0.8327\n",
            "Epoch 17/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 0.3274 - accuracy: 0.8694\n",
            "Epoch 18/40\n",
            "4/4 [==============================] - 0s 96ms/step - loss: 0.3024 - accuracy: 0.8531\n",
            "Epoch 19/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 0.3047 - accuracy: 0.8898\n",
            "Epoch 20/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 0.2939 - accuracy: 0.8653\n",
            "Epoch 21/40\n",
            "4/4 [==============================] - 0s 95ms/step - loss: 0.3151 - accuracy: 0.8694\n",
            "Epoch 22/40\n",
            "4/4 [==============================] - 0s 95ms/step - loss: 0.2886 - accuracy: 0.8898\n",
            "Epoch 23/40\n",
            "4/4 [==============================] - 0s 89ms/step - loss: 0.2739 - accuracy: 0.8694\n",
            "Epoch 24/40\n",
            "4/4 [==============================] - 0s 107ms/step - loss: 0.2733 - accuracy: 0.8898\n",
            "Epoch 25/40\n",
            "4/4 [==============================] - 0s 90ms/step - loss: 0.2390 - accuracy: 0.9020\n",
            "Epoch 26/40\n",
            "4/4 [==============================] - 0s 96ms/step - loss: 0.2391 - accuracy: 0.9143\n",
            "Epoch 27/40\n",
            "4/4 [==============================] - 0s 95ms/step - loss: 0.2439 - accuracy: 0.9143\n",
            "Epoch 28/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 0.2229 - accuracy: 0.9184\n",
            "Epoch 29/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 0.2114 - accuracy: 0.9184\n",
            "Epoch 30/40\n",
            "4/4 [==============================] - 0s 94ms/step - loss: 0.2135 - accuracy: 0.9143\n",
            "Epoch 31/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 0.2062 - accuracy: 0.9224\n",
            "Epoch 32/40\n",
            "4/4 [==============================] - 0s 96ms/step - loss: 0.1962 - accuracy: 0.9306\n",
            "Epoch 33/40\n",
            "4/4 [==============================] - 0s 89ms/step - loss: 0.1964 - accuracy: 0.9224\n",
            "Epoch 34/40\n",
            "4/4 [==============================] - 0s 93ms/step - loss: 0.2063 - accuracy: 0.9306\n",
            "Epoch 35/40\n",
            "4/4 [==============================] - 0s 97ms/step - loss: 0.1878 - accuracy: 0.9224\n",
            "Epoch 36/40\n",
            "4/4 [==============================] - 0s 87ms/step - loss: 0.1882 - accuracy: 0.9265\n",
            "Epoch 37/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 0.1849 - accuracy: 0.9224\n",
            "Epoch 38/40\n",
            "4/4 [==============================] - 0s 92ms/step - loss: 0.1854 - accuracy: 0.9429\n",
            "Epoch 39/40\n",
            "4/4 [==============================] - 0s 93ms/step - loss: 0.1722 - accuracy: 0.9347\n",
            "Epoch 40/40\n",
            "4/4 [==============================] - 1s 149ms/step - loss: 0.1633 - accuracy: 0.9388\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7b24150a8610>"
            ]
          },
          "metadata": {},
          "execution_count": 47
        }
      ],
      "source": [
        "cnn.fit(X_train, y_train, epochs = 40, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QNFWF72141JP",
        "outputId": "3de164b7-64af-4b01-dc32-cea03d451456"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2/2 [==============================] - 0s 16ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(X_val)\n",
        "y_pred_classes = np.round(pred).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0Jm6bHLH5MA1",
        "outputId": "0cdaa5e8-ee5a-4899-8786-19d287232e83"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9016393442622951, 0.896551724137931, 0.8030139935414424, 0.804750708270836)"
            ]
          },
          "metadata": {},
          "execution_count": 49
        }
      ],
      "source": [
        "accuracy_score(y_val, y_pred_classes), f1_score(y_val, y_pred_classes), cohen_kappa_score(y_val, y_pred_classes), matthews_corrcoef(y_val, y_pred_classes)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aCX03HuC5OWG",
        "outputId": "5294e05f-8a29-412b-ca01-c1ea933bcad7"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.8787878787878788, 0.9285714285714286)"
            ]
          },
          "metadata": {},
          "execution_count": 50
        }
      ],
      "source": [
        "cm1 = confusion_matrix(y_val, y_pred_classes)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "specificity, sensitivity"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "k1XklLHsCBy_"
      },
      "source": [
        "**Test**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0bzm03V7CJ2c"
      },
      "outputs": [],
      "source": [
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2zF9-4E7CDI9"
      },
      "outputs": [],
      "source": [
        "sample_size = xtrain.shape[0] # number of samples in train set\n",
        "time_steps  = xtrain.shape[1] # number of features in train set\n",
        "input_dimension = 1               # each feature is represented by 1 number"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "O0dDhydTCFpM"
      },
      "outputs": [],
      "source": [
        "train_data_reshaped = xtrain.values.reshape(sample_size,time_steps,input_dimension)\n",
        "n_timesteps = train_data_reshaped.shape[1]\n",
        "n_features  = train_data_reshaped.shape[2]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OpVNuOpMCHos"
      },
      "outputs": [],
      "source": [
        "cnn = Sequential()\n",
        "cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=2, activation='relu'))\n",
        "cnn.add(Conv1D(filters=128, kernel_size=2, activation='relu'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6ocTvVGtCV38"
      },
      "outputs": [],
      "source": [
        "cnn.add(MaxPool1D(pool_size=4))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PAHg1CwVCYzM"
      },
      "outputs": [],
      "source": [
        "cnn.add(Flatten())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "V3wk12xICcAs"
      },
      "outputs": [],
      "source": [
        "cnn.add(Dense(128, activation='relu'))\n",
        "cnn.add(Dense(64, activation='relu'))\n",
        "cnn.add(Dense(1, activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bOg1QfMpCfZE"
      },
      "outputs": [],
      "source": [
        "cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LmPqBaaWCiJc",
        "outputId": "172c6b83-3b3a-44b9-ff68-909c652d33de"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/40\n",
            "5/5 [==============================] - 2s 83ms/step - loss: 0.6914 - accuracy: 0.5359\n",
            "Epoch 2/40\n",
            "5/5 [==============================] - 0s 89ms/step - loss: 0.6822 - accuracy: 0.5359\n",
            "Epoch 3/40\n",
            "5/5 [==============================] - 0s 79ms/step - loss: 0.6664 - accuracy: 0.5359\n",
            "Epoch 4/40\n",
            "5/5 [==============================] - 0s 87ms/step - loss: 0.6355 - accuracy: 0.6503\n",
            "Epoch 5/40\n",
            "5/5 [==============================] - 0s 86ms/step - loss: 0.5895 - accuracy: 0.7059\n",
            "Epoch 6/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 0.5572 - accuracy: 0.7418\n",
            "Epoch 7/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.5477 - accuracy: 0.7092\n",
            "Epoch 8/40\n",
            "5/5 [==============================] - 0s 89ms/step - loss: 0.5075 - accuracy: 0.7549\n",
            "Epoch 9/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.4834 - accuracy: 0.7680\n",
            "Epoch 10/40\n",
            "5/5 [==============================] - 0s 91ms/step - loss: 0.4645 - accuracy: 0.7680\n",
            "Epoch 11/40\n",
            "5/5 [==============================] - 0s 79ms/step - loss: 0.4399 - accuracy: 0.7843\n",
            "Epoch 12/40\n",
            "5/5 [==============================] - 0s 89ms/step - loss: 0.4187 - accuracy: 0.7843\n",
            "Epoch 13/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.4119 - accuracy: 0.7941\n",
            "Epoch 14/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.3904 - accuracy: 0.8497\n",
            "Epoch 15/40\n",
            "5/5 [==============================] - 1s 143ms/step - loss: 0.3696 - accuracy: 0.8562\n",
            "Epoch 16/40\n",
            "5/5 [==============================] - 1s 141ms/step - loss: 0.3600 - accuracy: 0.8366\n",
            "Epoch 17/40\n",
            "5/5 [==============================] - 1s 143ms/step - loss: 0.3604 - accuracy: 0.8529\n",
            "Epoch 18/40\n",
            "5/5 [==============================] - 1s 142ms/step - loss: 0.4107 - accuracy: 0.8333\n",
            "Epoch 19/40\n",
            "5/5 [==============================] - 1s 125ms/step - loss: 0.3571 - accuracy: 0.8203\n",
            "Epoch 20/40\n",
            "5/5 [==============================] - 0s 87ms/step - loss: 0.3336 - accuracy: 0.9020\n",
            "Epoch 21/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 0.3115 - accuracy: 0.8529\n",
            "Epoch 22/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 0.3165 - accuracy: 0.8889\n",
            "Epoch 23/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 0.3036 - accuracy: 0.8725\n",
            "Epoch 24/40\n",
            "5/5 [==============================] - 0s 77ms/step - loss: 0.2925 - accuracy: 0.8954\n",
            "Epoch 25/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.2757 - accuracy: 0.9020\n",
            "Epoch 26/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.2600 - accuracy: 0.8922\n",
            "Epoch 27/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.2489 - accuracy: 0.9052\n",
            "Epoch 28/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 0.2439 - accuracy: 0.9020\n",
            "Epoch 29/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 0.2372 - accuracy: 0.9052\n",
            "Epoch 30/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.2358 - accuracy: 0.9183\n",
            "Epoch 31/40\n",
            "5/5 [==============================] - 0s 83ms/step - loss: 0.2488 - accuracy: 0.8791\n",
            "Epoch 32/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.2257 - accuracy: 0.9020\n",
            "Epoch 33/40\n",
            "5/5 [==============================] - 0s 85ms/step - loss: 0.2107 - accuracy: 0.9183\n",
            "Epoch 34/40\n",
            "5/5 [==============================] - 0s 79ms/step - loss: 0.2107 - accuracy: 0.9150\n",
            "Epoch 35/40\n",
            "5/5 [==============================] - 0s 79ms/step - loss: 0.1919 - accuracy: 0.9216\n",
            "Epoch 36/40\n",
            "5/5 [==============================] - 0s 80ms/step - loss: 0.1844 - accuracy: 0.9216\n",
            "Epoch 37/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 0.1682 - accuracy: 0.9248\n",
            "Epoch 38/40\n",
            "5/5 [==============================] - 0s 88ms/step - loss: 0.1620 - accuracy: 0.9314\n",
            "Epoch 39/40\n",
            "5/5 [==============================] - 0s 82ms/step - loss: 0.1634 - accuracy: 0.9216\n",
            "Epoch 40/40\n",
            "5/5 [==============================] - 0s 84ms/step - loss: 0.1601 - accuracy: 0.9412\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7b24150db190>"
            ]
          },
          "metadata": {},
          "execution_count": 59
        }
      ],
      "source": [
        "cnn.fit(xtrain, ytrain, epochs = 40, batch_size= 64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4NAgGEaVCmCE",
        "outputId": "b4569b4c-af37-4f6a-ea16-f6aaa8bb069e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5/5 [==============================] - 0s 11ms/step\n"
          ]
        }
      ],
      "source": [
        "pred = cnn.predict(xtest)\n",
        "pred = (pred > 0.5)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QGek9g4PCoT1",
        "outputId": "b862458c-9821-46ef-8a35-0872fa55b4a1"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9393939393939394,\n",
              " 0.9393939393939394,\n",
              " 0.8788990825688073,\n",
              " 0.8805147058823529)"
            ]
          },
          "metadata": {},
          "execution_count": 61
        }
      ],
      "source": [
        "accuracy_score(ytest, pred), f1_score(ytest, pred), cohen_kappa_score(ytest, pred), matthews_corrcoef(ytest, pred)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GcTwK6-kCr3V",
        "outputId": "6c0e0514-6c6e-4b87-94a1-4cb6a11a0755"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0.9117647058823529, 0.96875)"
            ]
          },
          "metadata": {},
          "execution_count": 62
        }
      ],
      "source": [
        "cm1 = confusion_matrix(ytest, pred)\n",
        "specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "specificity, sensitivity"
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

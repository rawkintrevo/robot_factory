__author__ = 'Trevor "Autogen" Grant'

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from datetime import datetime

def robot_college(learning_funcs, scorers, data):
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    output = list()
    for score in scorers:
        for lf in learning_funcs.keys():
            sp_time= datetime.now()
            print score[2], lf
            clf = GridSearchCV(learning_funcs[lf]['clf'], learning_funcs[lf]['tuned_parameters'], cv=5,
                               scoring=make_scorer(score[0], greater_is_better=score[1]), n_jobs=-1)
            clf.fit(X_train, y_train)
            y_true, y_pred = y_test, clf.predict(X_test)
            temp= {s[2]: s[0](y_true, y_pred) for s in scorers}
            # ^ yes, we calc scores for all here, but we are optimizing the clf against each scorer in different loops
            run_sec = float((datetime.now() - sp_time).microseconds) * 1e-6
            temp['params']= clf.best_params_
            temp['runtime']= run_sec
            temp['clf'] = lf
            temp['scorer'] = score[2]
            output.append(temp)
    return output


import numpy as np
def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    # Write code here
    X_train = np.asarray(X_train,dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test,dtype=float)
    classes = np.unique(y_train)
    prior ={}
    mean ={}
    vari = {}
    for c in classes:
        Xc = X_train[y_train==c]
        mean[c] = np.mean(Xc,axis=0)
        vari[c] = np.var(Xc,axis=0)+1e-9
        prior[c] = len(Xc)/len(X_train)
    predictions =[]
    for x in X_test:
        class_scores = []
        for c in classes:
            mean_c=mean[c]
            vari_c= vari[c]
            prior_c = prior[c]
            log_prior = np.log(prior_c)
            log_likelihood =np.sum((-0.5*np.log(2*np.pi*vari_c))-((x-mean_c)**2)/(2*vari_c))
            class_scores.append(log_prior+log_likelihood)
        predictions.append(classes[np.argmax(class_scores)])
    return predictions
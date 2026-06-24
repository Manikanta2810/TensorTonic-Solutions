import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    # Write code here
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    classes = np.unique(y_train)
    priors = {}
    probs = {}
    for c in classes:
        X_c = X_train[y_train == c]
        priors[c] = len(X_c) / len(X_train)
        n_c = len(X_c)
        probs[c] = (np.sum(X_c, axis=0) + 1) / (n_c + 2)

    predictions = []
    for x in X_test:
        classcores =[]
        for c in classes:
            p = probs[c]
            log_prior = np.log(priors[c])

            log_likelihood = np.sum(
                x * np.log(p + 1e-9)
                +
                (1 - x) * np.log(1 - p + 1e-9)
            )

            classcores.append(
                log_prior + log_likelihood
            )
        predictions.append(classcores)


    return predictions
            

   
        
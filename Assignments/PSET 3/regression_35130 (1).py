def regression_35130(Y, X):
    T = len(Y)
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    eps = Y - X @ B  # residuals
    RSS = (eps.T @ eps)  # Residual sum of squares
    RSSToT = (Y - np.mean(Y)).T @ (Y - np.mean(Y))
    sigsq = RSS / (T - 3)
    tstat = B / (np.sqrt(np.diag(np.linalg.inv(X.T @ X))) * np.sqrt(sigsq))
    R2 = 1 - RSS / RSSToT
    return B, tstat, R2

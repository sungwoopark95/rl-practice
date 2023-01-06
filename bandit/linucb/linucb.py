import numpy as np


## Disjoint LinUCB
class LinUCB:
    def __init__(self, arms, user_features, alpha=1.):
        self.alpha = alpha
        self.arms = arms # arm dataframe
        self.n_arms = arms.shape[0]
        self.user_features = user_features # list of column values
        self.A = [np.identity(len(user_features)) for _ in range(self.n_arms)]
        self.b = [np.zeros(shape=(len(user_features), 1)) for _ in range(self.n_arms)]
        
    def choose(self, x):
        # x: each row of the user dataframe
        user_feat = [x[feat] for feat in self.user_features]
        ps = np.zeros(shape=self.n_arms)
        for i in range(self.n_arms):
            # arm = dict(self.arms.iloc[i])
            # arm_feat = [arm[col] for col in self.arms.columns.values]
            # feat_vec = user_feat + arm_feat
            feat_vec = np.array(user_feat).reshape(-1, 1)
            theta_hat = np.linalg.inv(self.A[i]) @ self.b[i]
            p = (theta_hat.T@feat_vec) + (self.alpha*np.sqrt(feat_vec.T@np.linalg.inv(self.A[i])@feat_vec))
            ps[i] = p
        return np.argmax(ps)

    def update(self, x, a, r):
        user_feat = [x[feat] for feat in self.user_features]
        # arm = dict(self.arms.iloc[a])
        # arm_feat = [arm[col] for col in self.arms.columns.values]
        
        x_at = np.array(user_feat).reshape(-1, 1)
        self.A[a] += x_at @ x_at.T
        self.b[a] += r * x_at


## Hybrid LinUCB
class HybridLinUCB:
    def __init__(self, arms, user_features, alpha=1.):
        self.alpha = alpha
        self.arms = arms # arm dataframe
        self.n_arms = arms.shape[0]
        self.user_features = user_features  # user features + arm features -> length: d (column_values)
        self.k = len(user_features) * arms.shape[1]
        
        self.A0 = np.identity(self.k) # (k, k)
        self.b0 = np.zeros(shape=(self.k, 1)) # (k, 1)
        self.A = [np.identity(len(self.user_features)) for _ in range(self.n_arms)] # A_a, matrix for each arm, (d, d)
        self.B = [np.zeros(shape=(len(self.user_features), self.k)) for _ in range(self.n_arms)] # (d, k)
        self.b = [np.zeros(shape=(len(self.user_features), 1)) for _ in range(self.n_arms)] # (d, 1)
        
    def choose(self, x):
        ps = np.zeros(shape=self.n_arms)
        beta_hat = np.linalg.inv(self.A0) @ self.b0 # (k, 1)
        user_feat = [x[feat] for feat in self.user_features]
        for i in range(self.n_arms):
            A_a, B_a, b_a = self.A[i], self.B[i], self.b[i]
            arm = dict(self.arms.iloc[i])
            arm_feat = [arm[col] for col in self.arms.columns.values]
            # x_features = user_feat + arm_feat
            x_features = user_feat
            x_features = np.array(x_features).reshape(-1, 1)
            z_features = np.outer(user_feat, arm_feat).reshape(-1, 1)
            
            theta_hat = np.linalg.inv(A_a) @ (b_a - B_a@beta_hat) # (d, 1)
            
            # standard deviation
            s = ((z_features.T @ np.linalg.inv(self.A0) @ z_features)
               - (2 * z_features.T @ np.linalg.inv(self.A0) @ B_a.T @ np.linalg.inv(A_a) @ x_features)
               + (x_features.T @ np.linalg.inv(A_a) @ x_features)
               + (x_features.T @ np.linalg.inv(A_a) @ B_a @ np.linalg.inv(self.A0) @ B_a.T @ np.linalg.inv(A_a) @ x_features)
            )
            
            p = ((z_features.T @ beta_hat)
               + (x_features.T @ theta_hat)
               + (self.alpha * np.sqrt(s))
            )
            
            ps[i] = p
            
        return np.argmax(ps)
    
    def update(self, x, a, r):
        # a: index of the chosen action
        # r: real-valued reward
        user_feat = [x[feat] for feat in self.user_features]
        arm = dict(self.arms.iloc[a])
        arm_feat = [arm[col] for col in self.arms.columns.values]
        
        x_at = np.array(user_feat).reshape(-1, 1)        
        A_at, B_at, b_at = self.A[a], self.B[a], self.b[a]
        z_at = np.outer(user_feat, arm_feat).reshape(-1, 1)
        
        self.A0 += (B_at.T @ np.linalg.inv(A_at) @ B_at)
        self.b0 += (B_at.T @ np.linalg.inv(A_at) @ b_at)
        self.A[a] = A_at + (x_at @ x_at.T)
        self.B[a] = B_at + (x_at @ z_at.T)
        self.b[a] = b_at + r * x_at
        
        self.A0 += (z_at@z_at.T - self.B[a].T@np.linalg.inv(self.A[a])@self.B[a])
        self.b0 += (r*z_at - self.B[a].T@np.linalg.inv(self.A[a])@self.b[a])
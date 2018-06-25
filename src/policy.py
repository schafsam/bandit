import numpy as np

__author__ = "Samuel Schaffhauser"

class Recommender:

    def __init__(self):
        self.art_feat = {}
        self.As = {}
        self.AIs = {}
        self.bs = {}
        self.thetas = {}
        self.si = 31
        self.alpha = 0.2

        self.pointer = None
        self.xat = None


    def set_arms(self, arms):
        for action in arms:
            self.art_feat[action] = np.concatenate(
                (
                    np.ones(6,dtype=float),
                    np.tile(np.asarray(arms[action],dtype=float)[1:],5)
                )
            )
            self.As[action] = np.identity(self.si,dtype=float) + np.outer(self.art_feat[action],self.art_feat[action])
            self.AIs[action] = np.linalg.inv(self.As[action])
            self.bs[action] = np.zeros((self.si,),dtype=float)
            self.thetas[action] = np.zeros((self.si,),dtype=float)


    def update(self, reward):
        # sparse update policy
        if reward != -1:
                aux = self.xat * self.art_feat[self.pointer]
                self.As[self.pointer] = self.As[self.pointer] + np.outer(aux,aux)
                self.AIs[self.pointer] = np.linalg.inv(self.As[self.pointer])
                self.bs[self.pointer] = self.bs[self.pointer] + reward*aux
                self.thetas[self.pointer] = np.dot(self.AIs[self.pointer],self.bs[self.pointer])



    def recommend(self, time, articles, user_features):
        xa = []
        xa.extend(user_features)
        for i in range(5):
            xa.extend(user_features[1:])
        user_features = np.asarray(xa, dtype=float)
        p = {
            arm: float(
            np.dot(self.thetas[arm], user_features) + self.alpha *
            (np.dot(user_features, np.dot(self.AIs[arm], user_features))) ** 0.5)
             for arm in articles
            }
        self.pointer = max(p, key=p.get)
        self.xat = user_features
        return self.pointer

import numpy as np
import random
import operator
import pandas as pd

import matplotlib.pyplot as plt
import datetime
import time


# y1 is our prediction, y2 is true vector

y1=['david.marks@enron.com', 'roger.yang@enron.com', 'harumi.oyamada@enron.com', 'steve.kos@enron.com', 'piazzet@wharton.upenn.edu', 'jolly.jose@enron.com', 'gregg.lenart@enron.com', 'martha.benner@enron.com', 'brenda.whitehead@enron.com', 'frieda.schutza@enron.com']
y2=['richard.shapiro@enron.com', 'mark.whitt@enron.com', 'steven.j.kean@enron.com', 'shelley.corman@enron.com', 'rick.buy@enron.com', 'jarnold@enron.com', 'kimberly.watson@enron.com', 'jshankm@enron.com', 'e..haedicke@enron.com', 'jsteffe@enron.com']


def precision(y1, y2):
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nb_true_recipients = 0
    count = 0
    score = 0
    cardinal = 0

    if ((len(y1) != len(y2))&(len(y1)!=10)):
        print("Please check, vectors must be of same size")
    else:

        vector_lenghth = len(y1)

        for i in range(vector_lenghth):
            if ((y2[i] is not None) and (y2[i] != 0)):
                nb_true_recipients = nb_true_recipients + 1

            for j in range(len(y2)):
                if y1[i] == y2[j]:
                    v[i] = 1
                else:
                    pass

        for k in range(len(v)):
            if v[k] == 1:
                cardinal = cardinal + 1
                score = score + v[k] * cardinal / (k + 1)
            else:
                pass

        score = score / min(nb_true_recipients, len(y2))

        print("Vector : ", v)
        print("Score : ", score)

#mail_predicted=["A","B","C","D","E","F","G","H","I","J"]
#mail_true=["A","B","alpha","zongo","delta",0,0,0,0,0]


mail_predicted=['david.marks@enron.com', 'richard.shapiro@enron.com', 'harumi.oyamada@enron.com', 'steve.kos@enron.com', 'piazzet@wharton.upenn.edu', 'jolly.jose@enron.com', 'gregg.lenart@enron.com', 'martha.benner@enron.com', 'brenda.whitehead@enron.com', 'frieda.schutza@enron.com']
mail_true=['richard.shapiro@enron.com', 'mark.whitt@enron.com', 'steven.j.kean@enron.com', 'shelley.corman@enron.com', 'rick.buy@enron.com', 'jarnold@enron.com', 'kimberly.watson@enron.com', 'jshankm@enron.com', 'e..haedicke@enron.com', 'jsteffe@enron.com']

precision(mail_predicted,mail_true)


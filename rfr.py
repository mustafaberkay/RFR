import numpy as num
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

with open("Grand-slams-men-2013.csv") as f:
    csv_list = list(csv.reader(f))

fsp1_list = num.array([])
ace1_list = num.array([])
dbf1_list = num.array([])
wnr1_list = num.array([])
ufe1_list = num.array([])
bpc1_list = num.array([])
npa1_list = num.array([])
SRP = num.array([])
for row in csv_list:
    if row != csv_list[0]:
        fsp1_list = num.append(fsp1_list, int(row[6]))
        ace1_list = num.append(ace1_list, int(row[10]))
        dbf1_list = num.append(dbf1_list, int(row[11]))
        wnr1_list = num.append(wnr1_list, int(row[12]))
        ufe1_list = num.append(ufe1_list, int(row[13]))
        bpc1_list = num.append(bpc1_list, int(row[14]))
        npa1_list = num.append(npa1_list, int(row[16]))
        STsum = int(row[19])+int(row[20])+int(row[21])+int(row[22])+int(row[23])
        SRP = num.append(SRP, STsum)

ones = num.ones((1, len(fsp1_list)))
X = num.vstack((ones, fsp1_list, ace1_list, dbf1_list, wnr1_list, ufe1_list, bpc1_list, npa1_list)).T
Y = SRP.T

X_Train = X[0:200, :]
X_Test = X[200:, :]
Y_Train = Y[0:200]
Y_Test = Y[200:]

Y_pred_1 = num.array([])
Y_pred_2 = num.array([])
Y_pred_3 = num.array([])

m_squared_error_1 = num.array([])
m_squared_error_2 = num.array([])
m_squared_error_3 = num.array([])
N_est = num.array([])

for n_estimators in range(201):
    if n_estimators != 0:
        reg1 = RandomForestRegressor(max_depth=1,n_estimators=n_estimators,max_features="auto")
        reg2 = RandomForestRegressor(max_depth=1,n_estimators=n_estimators,max_features="sqrt")
        reg3 = RandomForestRegressor(max_depth=1,n_estimators=n_estimators,max_features=4)

        reg1.fit(X_Train,Y_Train)
        reg2.fit(X_Train, Y_Train)
        reg3.fit(X_Train, Y_Train)

        Y_pred_1 = reg1.predict(X_Test)
        Y_pred_2 = reg2.predict(X_Test)
        Y_pred_3 = reg3.predict(X_Test)

        m_squared_error_1 = num.append(m_squared_error_1, mean_squared_error(Y_Test, Y_pred_1))
        m_squared_error_2 = num.append(m_squared_error_2, mean_squared_error(Y_Test, Y_pred_2))
        m_squared_error_3 = num.append(m_squared_error_3, mean_squared_error(Y_Test, Y_pred_3))
        N_est = num.append(N_est, n_estimators)

Freg1 = RandomForestRegressor(max_depth=7, n_estimators=200, max_features=4)
Freg2 = RandomForestRegressor(max_depth=1, n_estimators=200, max_features=4)

Freg1.fit(X_Train,Y_Train)
Freg2.fit(X_Train, Y_Train)

Y_final_pred1 = Freg1.predict(X_Test)
Y_final_pred2 = Freg2.predict(X_Test)

plt.figure()
plt.plot(N_est, m_squared_error_1)
plt.legend("auto")
plt.plot(N_est, m_squared_error_2)
plt.legend("sqrt")
plt.plot(N_est, m_squared_error_3)
plt.legend("4")

plt. figure()

plt.scatter(Y_final_pred1, Y_Test - Y_final_pred1)
plt.scatter(Y_final_pred2, Y_Test - Y_final_pred2)

plt.show()

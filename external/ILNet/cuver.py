import matplotlib.pyplot as plt
import numpy as np

# result of teacher experiment
TPR_L = np.array([0.81115684, 0.80178738, 0.79135838, 0.78087658, 0.77003325, 0.75848382
, 0.74531929, 0.73243563, 0.71868506, 0.70462338, 0.69047273])
FPR_L = np.array([6.75362225e-05, 6.18346357e-05, 5.65192853e-05, 5.13510735e-05
, 4.65231166e-05, 4.24952212e-05, 3.86420521e-05, 3.51107460e-05
, 3.17173819e-05, 2.88298038e-05, 2.56663421e-05])
# result of student experiment
TPR_S2 = np.array([0.77000457, 0.75849527, 0.74701249, 0.73558202, 0.72390789, 0.71156923
, 0.69740233, 0.68381788, 0.67098209, 0.65639416, 0.6410997 ])

FPR_S2 = np.array([5.75216601e-05, 5.29419984e-05, 4.85554554e-05, 4.43436375e-05
, 4.06008215e-05, 3.69407705e-05, 3.34370532e-05, 3.02184152e-05
, 2.70641500e-05, 2.44616510e-05, 2.20890548e-05])

# result of student-kd experiment
# TPR_S3 = np.array([0.77772928, 0.76802378, 0.75788183, 0.74710613, 0.73598786, 0.72441239
# , 0.71269145, 0.70046342, 0.68803641, 0.67490285, 0.66214335])

# FPR_S3 = np.array([6.30577193e-05, 5.85608220e-05, 5.37052765e-05, 4.94750672e-05
# , 4.55483286e-05, 4.20170227e-05, 3.80351075e-05, 3.49911960e-05
# , 3.20484410e-05, 2.91976471e-05, 2.64572065e-05])

TPR_S3 = np.array([0.82522758,0.81404967,0.80206168,0.78867736,0.77633685,0.76352376
,0.74978699,0.73688148,0.72235499,0.70820258,0.69359933])

FPR_S3 = np.array([6.05103954e-05, 5.50846918e-05, 5.02291468e-05, 4.57046612e-05
, 4.18422959e-05, 3.80075191e-05, 3.43658599e-05, 3.10644567e-05
, 2.82136629e-05, 2.55835757e-05, 2.32477641e-05])

# draw scatter and line 
plt.scatter(FPR_L, TPR_L, marker='o')
plt.scatter(FPR_S2, TPR_S2, marker='*')
plt.scatter(FPR_S3, TPR_S3, marker='^')
plt.plot(FPR_L, TPR_L, label='teacher')
plt.plot(FPR_S2, TPR_S2, label='student')
plt.plot(FPR_S3, TPR_S3, label='student-kd')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve of ILNet: teacher L vs student S, dataset: IRSTD-1k, traindata: IRSTD-1k1', fontsize=10)
plt.grid()
plt.savefig('ROC_ILNet_IRSTD-1k.png')
plt.show()





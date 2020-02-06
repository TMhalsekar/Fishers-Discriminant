trY = np.transpose(trY)
df['final'] = trY
mean = pd.DataFrame(columns=df['final'])
for index, row in df.groupby('final'):
    row = row.drop(columns=['final'])
    mean[index] = row_data.mean()
sc_m = np.zeros((10, 10))
for index, row_data in df.groupby('final'):
    row_data = row_data.drop(['final'], axis=1)
    s = np.zeros((10, 10))
    for p, r in row.iterrows():
        x= r.values.reshape(10, 1)
        m = mean[index].values.reshape(10, 1)
        s += (x - m).dot((x - m).T)
    sc_m += s
f_m = df.drop(columns=['final']).mean()
bsc_matrix = np.zeros((10, 10))
for index in means:
    n = len(df.loc[df['final'] == index].index)
    mc, m = means[index].values.reshape(10, 1), feature_means.values.reshape(10, 1)
    bsc += n * (mc - m).dot((mc - m).T)
eigen_val, eigen_vec = np.linalg.eig(
    np.linalg.inv(sc_m).dot(bsc_m))
tuples = [(np.abs(eigen_val[i]), eigen_vec[:, i]) for i in range(len(eigen_val))]
tuples = sorted(pairs, key=l x: x[0], reverse=True)
wmatrix = np.hstack((pairs[0][1].reshape(10, 1))).real
print(w_matrix)
df2 = df.drop(columns = ['final'])
xlda = np.array(df2.dot(wmatrix))

df['x_lda'] = xlda
m1 =xlda[0:200].mean()
m2 =xlda[200:400].mean()
print(m1, m2)
b = (m1 + m2)/2
df['predict'] = df['x_lda'].apply(l x: 8 if x<0 else 5)
class =  df[df['predict'] == df['final']].shape[0]
print("Training accuracy is:")
print(float(class) / 100)
print(float(class)/400)
df3 = pd.DataFrame(tsX.T)
df4 = pd.DataFrame(pca.transform(df_test))
xlda = np.array(df_test1.dot(w_matrix))
df_test1['xldatest'] = xldatest
df3['final'] = tsY.T
df3['predict'] = df3['xldatest'].apply(lambda x: 8 if x < 0 else 5)
clasf = df3[df3['predict'] == df3['final']].shape[0]
print("Testing accuracy is:")
print(float(correct_classification) / 100)

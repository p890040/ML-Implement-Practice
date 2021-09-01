import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#%% Initial data
K = 3
# x = np.linspace(-1, 1)
# y = x + np.random.normal(size=x.size)
x = np.array([-1.000000000000000000e+00,-9.589843750000000000e-01,-9.184570312500000000e-01,-8.774414062500000000e-01,-8.369140625000000000e-01,-7.958984375000000000e-01,-7.548828125000000000e-01,-7.143554687500000000e-01,-6.733398437500000000e-01,-6.328125000000000000e-01,-5.917968750000000000e-01,-5.507812500000000000e-01,-5.102539062500000000e-01,-4.694824218750000000e-01,-4.284667968750000000e-01,-3.876953125000000000e-01,-3.469238281250000000e-01,-3.061523437500000000e-01,-2.653808593750000000e-01,-2.244873046875000000e-01,-1.837158203125000000e-01,-1.428222656250000000e-01,-1.020507812500000000e-01,-6.121826171875000000e-02,-2.040100097656250000e-02,2.040100097656250000e-02,6.121826171875000000e-02,1.020507812500000000e-01,1.428222656250000000e-01,1.837158203125000000e-01,2.244873046875000000e-01,2.653808593750000000e-01,3.061523437500000000e-01,3.469238281250000000e-01,3.876953125000000000e-01,4.284667968750000000e-01,4.694824218750000000e-01,5.102539062500000000e-01,5.507812500000000000e-01,5.917968750000000000e-01,6.328125000000000000e-01,6.733398437500000000e-01,7.143554687500000000e-01,7.548828125000000000e-01,7.958984375000000000e-01,8.369140625000000000e-01,8.774414062500000000e-01,9.184570312500000000e-01,9.589843750000000000e-01,1.000000000000000000e+00])
y = np.array([-1.651367187500000000e+00,-2.632812500000000000e+00,-9.619140625000000000e-01,-3.167968750000000000e+00,-8.178710937500000000e-01,-1.077148437500000000e+00,-1.008789062500000000e+00,1.414062500000000000e+00,-8.349609375000000000e-01,-1.677734375000000000e+00,7.099609375000000000e-01,-5.903320312500000000e-01,-1.496093750000000000e+00,-2.128906250000000000e-01,-2.109375000000000000e+00,-2.810058593750000000e-01,-1.000000000000000000e+00,-1.601562500000000000e+00,1.755859375000000000e+00,-6.381835937500000000e-01,-4.985351562500000000e-01,-2.205078125000000000e+00,6.367187500000000000e-01,1.662109375000000000e+00,2.067871093750000000e-01,3.179687500000000000e+00,2.070312500000000000e-01,4.052734375000000000e-01,3.984375000000000000e-01,6.445312500000000000e-01,-5.444335937500000000e-01,6.738281250000000000e-01,6.176757812500000000e-01,4.565429687500000000e-01,3.281250000000000000e-01,1.244140625000000000e+00,1.085937500000000000e+00,1.816406250000000000e+00,5.522460937500000000e-01,-2.875976562500000000e-01,6.884765625000000000e-02,-1.357421875000000000e-01,6.977539062500000000e-01,1.665039062500000000e-01,3.261718750000000000e-01,-1.440429687500000000e-01,-5.175781250000000000e-02,-1.318359375000000000e+00,-7.128906250000000000e-02,1.732421875000000000e+00])
# center_x = np.random.uniform(low=x.min(), high=x.max(), size=3)
# center_y = np.random.uniform(low=y.min(), high=y.max(), size=3)
center_x = np.array([0.01, -0.264, 0.547])
center_y = np.array([1.04, -0.286, 1.015])
K_COLOR = ['red', 'green', 'blue']
K_category = ['s', 'x', 'v']

fig = plt.figure()
ax = fig.gca()
ax.scatter(x, y, s=8, c='black')
for i in range(K):
    ax.scatter(center_x[i], center_y[i], s=80, marker=K_category[i], c=K_COLOR[i]) 

#%% K-means function
def K_means(points, center_points, K):
    fig = plt.figure()
    ax = fig.gca()
    x, y = points
    center_x, center_y = center_points
    
    K_distance = np.zeros((K, x.size))
    for idx_p in range(x.size):
        for idx_c in range(K):
            dis = np.sqrt((center_x[idx_c] - x[idx_p])**2 + (center_y[idx_c] - y[idx_p])**2)
            K_distance[idx_c, idx_p]=dis

    _clusters = np.argmin(K_distance, axis=0)
    clusters = [np.where(_clusters==i)[0] for i in range(K)]    
    new_center_x, new_center_y = np.zeros_like(center_x), np.zeros_like(center_y)
    center_mv = []
    for i in range(K):
        new_center_x[i] = np.mean(x[clusters[i]])
        new_center_y[i] = np.mean(y[clusters[i]])
        center_mv.append(np.sqrt((new_center_x[i] - center_x[i])**2 + (new_center_y[i] - center_y[i])**2))
        ax.scatter(center_x[i], center_y[i], s=80, marker=K_category[i], c=K_COLOR[i], alpha=0.1) 
        ax.scatter(x[clusters[i]], y[clusters[i]], s=8, c=K_COLOR[i])
        ax.scatter(new_center_x[i], new_center_y[i], s=80, marker=K_category[i], c=K_COLOR[i]) 

    return new_center_x, new_center_y, sum(center_mv)
    
#%% Run the code
step = 10
center_x_his, center_y_his = [], [] 
new_center_x, new_center_y = center_x, center_y
for s in range(step):
    center_x_his.append(new_center_x)
    center_y_his.append(new_center_y)
    new_center_x, new_center_y, mv = K_means(points=[x, y], center_points=[new_center_x, new_center_y], K=K)
    print(f'Step {i} : moving distance {mv:.3f}')
    if(mv < 0.05):
        print(f'[Early stopped by moving distance : {mv:.3f}')
        break

fig = plt.figure()
ax = fig.gca()
ax.scatter(x, y, s=8, c='black')
for i in range(s):
    for k in range(K):
        ax.scatter(center_x_his[i][k], center_y_his[i][k], s=80, marker=K_category[k], c=K_COLOR[k], alpha=((i+1)/s)) 



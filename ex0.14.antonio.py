import numpy as np
import matplotlib.pyplot as plt
import lxmls.readers.galton as galton

def get_error_2dim(w):
    y = galton_data[:, 1]  # sons
    x = galton_data[:, 0]  # fathers
    # print w
    return np.sum((x * w[1] + w[0] - y) ** 2)


def get_2dim_error_grad(w):
    y = galton_data[:, 1]  # sons
    x = galton_data[:, 0]  # fathers
    g0 = np.sum(2 * (x * w[1] + w[0] - y))
    g1 = np.sum(2 * x * ( x * w[1] + w[0] - y))
    return np.array([g0, g1])

def grad_desc2D(start_x, func, grad):
    precision = 0.00001
    step_size0 = 0.0001
    step_size1 = 0.0000001

    max_it = 50000
    x_new = start_x
    res = []
    for i in xrange(max_it):
        x_old = x_new
        grads = grad(x_new)
        x_new = np.array([ x_old[0] - step_size0 * grads[0],x_old[1] - step_size1 * grads[1]])
        #print 'grad0: %(f)d   grad1: %(s)d' % {'f': grad(x_new[0])[0], 's': grad(x_new[1])[1]}
        f_x_new = func(x_new)
        f_x_old = func(x_old)
        print "weights=", x_old, x_new, "grads=", grads, "fx=", f_x_old, f_x_new

        res.append([x_new, f_x_new])
        if (abs(f_x_new - f_x_old) < precision):
            print "change in func too small, leaving"
            return np.array(res)

    print "exceeded max iterations, leaving"
    return np.array(res)

galton_data = galton.load()

res =grad_desc2D(np.array([0, 0]), get_error_2dim, get_2dim_error_grad)
print "solution estimated by gradient descent:"
print res[-1]



#check solution with numeric method
y = galton_data[:, 1]  # sons
x = galton_data[:, 0]  # fathers
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print "numerically calculated solution:"
print(c,m)
print "error of numerically computed solution:"
print get_error_2dim([c,m])



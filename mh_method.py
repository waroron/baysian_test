import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm, gamma


def metropolice_hasting(n, q, q_rvs, f):
    """
    Metropolice-Hasting法を用いて，目的分布fに従う乱数aをサンプリングする
    :param q: 提案分布
    :param f: 事後分布
    :return:
    """
    sumple = []
    theta = np.random.randint(0, 1)
    for num in range(n):
        # a = norm.rvs(loc=1, scale=0.5)
        a = q_rvs()
        tmp1 = q(a) * f(theta)
        tmp2 = q(theta) * f(a)
        if tmp1 > tmp2:
            r = tmp2 / tmp1
            if r > np.random.rand():
                sumple.append(a)
                theta = a
        else:
            sumple.append(a)
            theta = a

    return np.array(sumple)


def random_walk_mh(n, q_rvs, f):
    sumple = []
    theta = np.random.randint(0, 1)
    for num in range(n):
        a = q_rvs()
        e = np.random.uniform(-0.2, 0.2)
        a += e
        tmp1 = f(theta)
        tmp2 = f(a)
        if tmp1 > tmp2:
            r = tmp2 / tmp1
            if r > np.random.rand():
                sumple.append(a)
                theta = a
        else:
            sumple.append(a)
            theta = a

    return np.array(sumple)


LOC = 1
SCALE = 2.0


def q(x):
    return norm.pdf(x, loc=LOC, scale=SCALE)


def q_rvs():
    return norm.rvs(loc=LOC, scale=SCALE)


def k_fg(theta):
    return gamma.pdf(theta, 11, scale=1./13)


if __name__ == '__main__':
    n_burn_in = 30000
    # theta = metropolice_hasting(n_burn_in, q, q_rvs, k_fg)
    theta = random_walk_mh(n_burn_in, q_rvs, k_fg)

    # plot samples
    imgs = []
    fig = plt.figure()
    length = len(theta)
    plt.figure(figsize=(5, 5))
    plt.title("Histgram by Independent Metropolice-Hasting Method")
    plt.hist(theta[:], bins=50, normed=True, histtype='stepfilled', alpha=0.2)
    xx = np.linspace(0, 2.5, 501)
    img = plt.plot(xx, gamma(11.0, 0.0, 1 / 13.).pdf(xx))
    imgs.append(img)

    # plt.savefig('mh_fig.png')
    plt.savefig('random_walk_fig.png')

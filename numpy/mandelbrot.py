import numpy as np
import matplotlib.pyplot as plt
'''
曼德勃罗集合（Mandelbrot Set）或曼德勃罗复数集合，是一种在复平面上组成分形的点的集合，因由曼德勃罗提出而得名。
曼德博集合可以使复二次多项式fc(z)=z²+c进行迭代来获得。其中，c是一个复参数。对于每一个c，从 z = 0 开始对fc(z)进行迭代。
序列(0,fc(0),fc(dc(0)...)的值或者延伸到无限大，或者只停留在有限半径的圆盘内（这与不同的参数c有关）。曼德布洛特集合就是使以上序列不延伸至无限大的所有c点的集合。
https://blog.csdn.net/baimafujinji/article/details/50859174
'''


def mandelbrot(h, w, maxit=50):
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2         # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much
    return divtime


plt.imshow(mandelbrot(500, 500))
plt.show()
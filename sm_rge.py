import math
import numpy as np

pi2 = math.pi**2
pi4 = math.pi**4

class rge:
    def __init__(self,twoLoop=False):
        self.twoLoop = twoLoop
    def f(self,y,t):
        g1 = y[0]
        g2 = y[1]
        g3 = y[2]
        yt = y[3]
        yb = y[4]
        ytau = y[5]
        Lambda1 = y[6]
        Zh = y[7]
        mu12 = y[8]
  
        dydt = np.zeros(len(y))

        # g1 RGE
        dydt[0] = ((41*np.power(g1,3))/10.)/(16.*pi2) 

        # g2 RGE
        dydt[1] = ((-19*np.power(g2,3))/6.)/(16.*pi2) 

        # g3 RGE
        dydt[2] = (-7*np.power(g3,3))/(16.*pi2)

        # yt RGE
        dydt[3] = ((-17*np.power(g1,2)*yt)/320. - (9*np.power(g2,2)*yt)/64. - \
(np.power(g3,2)*yt)/2. + (3*np.power(yb,2)*yt)/32. + (9*np.power(yt,3))/32. + \
(yt*np.power(ytau,2))/16.)/pi2 

        # yb RGE
        dydt[4] = (-(np.power(g1,2)*yb)/64. - (9*np.power(g2,2)*yb)/64. - \
(np.power(g3,2)*yb)/2. + (9*np.power(yb,3))/32. + (3*yb*np.power(yt,2))/32. + \
(yb*np.power(ytau,2))/16.)/pi2

        # ytau RGE
        dydt[5] = ((-9*np.power(g1,2)*ytau)/64. - (9*np.power(g2,2)*ytau)/64. + \
(3*np.power(yb,2)*ytau)/16. + (3*np.power(yt,2)*ytau)/16. + \
(5*np.power(ytau,3))/32.)/pi2 

        # Lambda1 RGE
        dydt[6] = ((27*np.power(g1,4))/3200. + \
(9*np.power(g1,2)*np.power(g2,2))/320. + (9*np.power(g2,4))/128. - \
(9*np.power(g1,2)*Lambda1)/80. - (9*np.power(g2,2)*Lambda1)/16. + \
3*np.power(Lambda1,2)/2.  + (3*Lambda1*np.power(yb,2))/4. \
- (3*np.power(yb,4))/8. + (3*Lambda1*np.power(yt,2))/4. - (3*np.power(yt,4))/8. \
+ (Lambda1*np.power(ytau,2))/4. - np.power(ytau,4)/8.)/pi2


        # Z_h RGE
        dydt[7] = ((-9*np.power(g1,2))/320. - (9*np.power(g2,2))/64. + \
(3*np.power(yb,2))/16. + (3*np.power(yt,2))/16. + np.power(ytau,2)/16.)/pi2 


        # mu_1^2 RGE
        dydt[8] = -(((9*np.power(g1,2)*mu12)/160. + \
(9*np.power(g2,2)*mu12)/32. - (3*Lambda1*mu12)/4. \
- (3*mu12*np.power(yb,2))/8. - (3*mu12*np.power(yt,2))/8. - \
(mu12*np.power(ytau,2))/8.)/pi2)

        return dydt


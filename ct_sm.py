import math
import numpy as np
import sys
from cosmoTransitions import generic_potential
import sm_rge
from scipy import optimize
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
       
v2 = 246.**2

   
def devFromInput(x, mh, vev, model):
    """
    Compute the deviation in v, mh
    """

    #mu12, mu22, l1, l2, l3, l4, l5 = x
    mu12, l1 = x
    
    v = vev
    # Modify RGE initial conditions
    y0 = model.y0
    y0[6] = l1
    y0[8] = mu12
    # update the input values stored in the model object!
    model.y0 = y0
    inputScale = np.sqrt(model.inputRgScaleSq)
    # Integrate RGEs to mu=v
    tmin = 0; tmax = np.log(v/inputScale)
    tvev = np.log(v/inputScale)
    #ti = np.linspace(tmin,tmax)
    ti = np.array([0., tvev])
    model.rgsys.twoLoop = True
    y_at_t = integrate.odeint(model.rgsys.f,y0,ti)
    # y_at_t: [0] initial, [1] at vev
    y_at_v = y_at_t[1]
    # Update the potential parameters, compute v and mh
    model.renormScaleSq = v**2.
    g1,g2,g3,yt,yb,ytau,l1,Zh,mu12 = y_at_v

    model.setPara(l1,mu12)

    model.setGauge(g1,g2,g3)
    model.setYukawa(yt,yb,ytau)

    model.setFieldRenorm(Zh)
    
    mh2l = model.d2V([v],T=0)[0,0]

    f = model.gradV([v],T=0.)[0]/v**3
    if mh2l > 0.:
      g = (mh2l - mh**2)/mh**2
    else: 
      g = 100.

    return [f, g]

def findParaFull(model, mh):
    """
    Find l1 and mu12 at input scale that correspond to 
    dV(h=v) =0 and d2V(v) = mh^2 at rgScale = vev

    """ 
    vev = 246.22
    lam1i = model.l1;
    mu12i = model.mu12;

    func = lambda x: devFromInput(x, mh, vev, model)
    #sol = scipy.optimize.newton_krylov(func, [lam1_init,mu12_init], f_tol=1e-3)
    x_guess = [mu12i, lam1i]
    print "Deviations from desired values before optimization:", func(x_guess)
    sol = optimize.fsolve(func, x_guess)
    print "Deviations from desired values after optimization:", func(sol)
    # Solve model RGEs, now with the updated initial conditions
    model.solve_rge(model.y0)
    return sol


class sm_eft(generic_potential.generic_potential):
    """
    """
    def setPara(self, l1, mu12):
      self.l1 = l1
      self.mu12 = mu12

    def setYukawa(self, yt, yb, ytau):
      self.yt = yt
      self.yb = yb
      self.ytau = ytau
    
    def setGauge(self, g1, g2, g3):
      # g1 from RGEs is in GUT convention
      self.g1 = g1
      self.g2 = g2
      self.g3 = g3

    def setFieldRenorm(self, Zh):
      self.Zh = Zh

    def init(self, mh=125.,inputRgScale=91.1887):
        """
        """
        # The init method is called by the generic_potential class, after it 
        # already does some of its own initialization in the default __init__() 
        # method. This is necessary for all subclasses to implement.

        # This first line is absolutely essential in all subclasses. 
        # It specifies the number of field-dimensions in the theory.
        self.Ndim = 1
        

        # The field independent piece of the potential. Will initialize properly later. 
        self.Omega = 0.
        
        # This next block sets all of the parameters that go into the potential 
        # and the masses. This will obviously need to be changed for different 
        # models.
        g1 = np.sqrt(5./3.)*3.47007677e-1; g2 = 6.47921668e-1; g3 = 1.21978;
        yt = 9.97858113e-1; yb = 3.48346352E-02; ytau = 1.01055057e-2;
        Zh0 = 1.; 



        # Guesses for the couplings that will give the proper values of mh and v at the vev scale
        v = 246.22
        mu12 = 88.6**2.
        l1 = 0.11 
        print "Guesses for mu12 and l1 = ", mu12, l1

        self.setPara(l1,mu12)

        self.setGauge(g1,g2,g3)
        self.setYukawa(yt,yb,ytau)

        self.setFieldRenorm(Zh0)

	    # Set up the RGEs
        self.rgsys = sm_rge.rge()
        # self.renormScaleSq is the renormalization scale used in the 
        # Coleman-Weinberg potential.
        #self.renormScaleSq = v2
        self.inputRgScaleSq = inputRgScale**2
        self.renormScaleSq = self.inputRgScaleSq 
        # Initial conditions for RG running

        # Include the daisy corrections or not
        self.daisyResum = True; 


        # Run between tmin and tmax
        print "RG running with initial parameters at the input scale..."
        self.tmin = 0.; self.tmax = 30.; self.dt = 0.1;
        # Update the inpute scale parameters with the new values
        self.y0 = np.array([g1,g2,g3,yt,yb,ytau,l1,Zh0,mu12])
        #self.solve_rge(self.y0)

        #print "Setting rg scale to 246.22 GeV..."
        #self.set_rg_scale(v)
      
        print "Finding the correct parameters at the input scale..."
        # Note that this function recomputes the rges, and changes y0
        findParaFull(self, mh)
        # Verify that everything went well...
        print "Final values of mu12, l1 = ", self.mu12, self.l1
        print "Scale set at mu = ", np.sqrt(self.renormScaleSq)
        vev = optimize.minimize(lambda x: self.Vtot(x,T=0),[200])['x']
        print "vev = ", vev 
        print "mh = ", np.sqrt(self.d2V(vev,T=0)[0,0])


    def solve_rge(self, y0):
      # Solve for the running parameters as a function of t = ln(mu/mu0)
      ti = np.arange(self.tmin,self.tmax,self.dt)
      self.rgsys.twoLoop = True
      y_at_t = integrate.odeint(self.rgsys.f,y0,ti)
      self.check_perturbativity(ti,y_at_t)
      # Create interpolating functions so we can evaluate parameters at any t
      self.y_at_t = [interpolate.interp1d(ti,y_at_t[:,i]) for i in range(0,len(y_at_t[0]))]

    def check_perturbativity(self, ti, yi):
      for t, y in zip(ti,yi):
        if np.any(np.fabs(y[:7]) > 4.*math.pi):
          print "Perturbativity violated at mu = ", np.sqrt(self.inputRgScaleSq)*np.exp(t)
          return

    def set_rg_scale(self, mu):
      t = np.log(mu/np.sqrt(self.inputRgScaleSq))
      assert t >= self.tmin and t < self.tmax

      self.renormScaleSq = mu**2 

      y = [f(t) for f in self.y_at_t]
     
      g1,g2,g3,yt,yb,ytau,l1,Zh,mu12 = y

      self.setPara(l1, mu12)

      self.setGauge(g1,g2,g3)
      self.setYukawa(yt,yb,ytau)

      self.setFieldRenorm(Zh)
      #self.setOmega()

      
    def plot_rg_solutions(self, tmin=0, tmax=30):
       
      ti = np.arange(tmin,tmax,self.dt)
      yi = [f(ti) for f in self.y_at_t] 
      plt.ylabel(R"$g_i$",fontsize=16)
      plt.plot(ti,yi[0],ti,yi[1],ti,yi[2],linewidth=2.0)
      plt.show()
      plt.ylabel(R"$y_i$",fontsize=16)
      plt.plot(ti,yi[3],ti,yi[4],ti,yi[5],linewidth=2.0)
      plt.show()
      plt.ylabel(R"$\lambda_1$",fontsize=16)
      plt.plot(ti,yi[6],linewidth=2.0)
      plt.show()
      plt.ylabel(R"$Z_h$",fontsize=16)
      plt.plot(ti,yi[7],linewidth=2.0)
      plt.show()
      plt.ylabel(R"$\mu_1^2$",fontsize=16)
      plt.plot(ti,yi[8],linewidth=2.0)
      plt.show()


    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is useful to set if there is, for example, a Z2 symmetry
        in the theory and you don't want to double-count all of the phases. In 
        this case, we're throwing away all phases whose zeroth (since python 
        starts arrays at 0) field component of the vev goes below -5. Note that 
        we don't want to set this to just going below zero, since we are
        interested in phases with vevs exactly at 0, and floating point numbers 
        will never be accurate enough to ensure that these aren't slightly 
        negative.
        """
        return (np.array([X])[...,0] < -5.0).any()
        
                        
    def V0(self, X):
        """
        This method defines the tree-level potential. It should generally be 
        subclassed. (You could also subclass Vtot() directly, and put in all of 
        quantum corrections yourself).
        """
        X = np.asanyarray(X)

        v = X[...,0]
        v = self.Zh*v;
        r = (2*self.mu12*(v**2) + self.l1*(v**4) )/4.
        return r
        
    def boson_massSq(self, X, T):
        X = np.array(X)
        #vArr,phiArr = X[...,0], X[...,1]
        v = X[...,0]

        v = self.Zh*v

        # U(1)_Y coupling in the RG equations is given in the GUT normalization, convert back
        gp = np.sqrt(3./5.)*self.g1
        g = self.g2
       
        mW2 = np.array([0.25*(g**2)*(v**2 )])
        mZ2 = np.array([0.25*(g**2 + gp**2)*(v**2)])


        if self.daisyResum:
          # Temperature corrections for scalars
          c1 = g**2/8. + (g**2 + gp**2)/16. \
             + self.l1/2. + self.yt**2/4. + self.yb**2/4. + self.ytau**2/12.
          #c1 = c2 = 0
          MSqEven = np.array([self.mu12 + c1*T**2 + 3.*self.l1*v**2 ])
          MSqOdd = np.array([self.mu12 + c1*T**2 + self.l1*v**2 ])
          MSqCharged = np.array([self.mu12 + c1*T**2 + self.l1*v**2 ])



          # Temperature corrections for longitudinal gauge bosons
          mW2L = mW2 + (11/6.)*g*g*T*T

          Discr = np.sqrt( (mZ2/2. + (11./12.)*(g**2 + gp**2)*T**2)**2 \
                         - (g**2)*(gp**2)*((11./6.)*T**2)*(0.5*v**2 + (11./6.)*T**2) )
          mZ2L = mZ2/2. + (g**2 + gp**2)*T**2 + Discr
          mA2L = mZ2/2. + (g**2 + gp**2)*T**2 - Discr
          
          #mZ2L = mZ2;
          #mA2L = np.array([0.]);
          #mW2L = mW2;


          """

          # Turn on and off various contributions to check what is driving the transition
          if T > 0:
            # Get rid of gauge bosons
            #dof = np.array([1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0])
            # Get rid of A, Hpm
            #dof = np.array([1, 1, 1, 0, 2, 0, 2, 1, 1, 4, 2])
            # Get rid of h, H
            dof = np.array([0, 0, 1, 1, 2, 2, 2, 1, 1, 4, 2])

          else:
            dof = np.array([1, 1, 1, 1, 2, 2, 2, 1, 1, 4, 2])
          """
          #print (MSqEven, MSqOdd, MSqCharged,mZ2,mZ2L,mA2L, mW2, mW2L)
          M = np.concatenate((MSqEven, MSqOdd, MSqCharged,mZ2,mZ2L,mA2L, mW2, mW2L))
          M = np.rollaxis(M, 0, len(M.shape))
          dof = np.array([1, 1, 2, 2, 1, 1, 4, 2])
          # c_i = 1/2 for tranverse, 3/2 for longitudinal
          c = np.array([1.5, 1.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5])
        return M, dof, c

    def fermion_massSq(self, X):
        X = np.array(X)
        v = X[...,0]

        v = self.Zh*v; 

        mt2 = 0.5*(self.yt*v)**2
        mb2 = 0.5*(self.yb*v)**2
        mtau2 = 0.5*(self.ytau*v)**2
        M = np.array([mt2, mb2, mtau2])
        M = np.rollaxis(M, 0, len(M.shape))
        
        dof = np.array([12, 12, 4])


        c = np.array([1.5, 1.5, 1.5])
        return M, dof

    def gaugeInvariantVeffMin(self, X0, T):
      """ Gauge invariant value of the potential at an extremum using the method of 1101.4665.
          The tree and one loop potential are both evaluated at the tree-level critical point. 
          For expample, for the EW breaking min, this is just -\mu_1^2/\lambda_1.
          
          X0 - value of field at the tree-level extremum (array). 
          T - temperature
      """
      X0min = optimize.minimize(lambda x: self.V0(x),X0)['x']
      return self.DVtot(X0min,T)

    def calcGaugeInvariantTc(self):
      """ Solve for the gauge invariant critical temperature where V(0,T=Tc)  = V(EW, T=Tc) 
          using the prescription of 1101.4665.
      """
      X0 = np.array([np.sqrt(-self.mu12/self.l1),0])
      Tmax = 0.;
      while self.gaugeInvariantVeffMin(X0,Tmax) < 0:
        Tmax = Tmax + 10.

      Tc = optimize.brentq(lambda T: self.gaugeInvariantVeffMin(X0,T),0.,Tmax)
      return Tc

    def calcGaugeInvariantVev(self,T):
      vev0 = 246.22
      mh = 126.
      gp = np.sqrt(3./5.)*self.g1
      g = self.g2

      c1 = g**2/8. + (g**2 + gp**2)/16. \
             + self.l1/2. + self.lL/12. + self.lS/12. + self.l3/12. \
             + self.yt**2/4. + self.yb**2/4. + self.ytau**2/12.
      T02 = (mh**2)/(2.*c1)
      vbar2 = vev0*vev0*(1.-T*T/T02)
      if vbar2 > 0:
        return np.sqrt(vbar2)
      else:
        return 0. 

    def calcGaugeInvariantStrength(self):
      Tc = self.calcGaugeInvariantTc()
      vbar = self.calcGaugeInvariantVev(Tc)
      return vbar/Tc
      
    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model, 
        # and we want to include both of them.
        v = v2**.5
        return [np.array([v])]
        

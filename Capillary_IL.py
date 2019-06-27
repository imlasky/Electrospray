import numpy as np
from common_trunc import multiDim,rename,sumOverNs,tooManySites
from common_trunc import EP0,CHARGE_E,PLANCK,BOLTZ,GRAV

class Model(object):

    PROPELLANTS_IL = 'All'
    PROPELLANTS_LM = ''
    SUBSTRATES_CHANNEL = 'All'
    SUBSTRATES_POROUS = ''

    '''
    Supply values here or use "_form_|<datatype>|<label>|<initialValue>" to
    define a form field to be automatically generated in the web app. When
    testing in standalone mode (no web app), no form is created and instead
    the initial values are used. When the initial value must be read from the
    database, use <initialValue> = tableName.columnName.

    All entries defined as an entry pair, <key> and <key>_std, are
    created as random variables.
    '''
    X = {
         'Ne': '_form_|int|Number of Emitters|127',
         'Nspe': 1,
         'Polarity':'Positive',

         'Independent Variable':'V', # V, T or P
         'Vmin':700,'Vmax':1850,'V':1700, # min, max, nominal
         'Tmin':250,'Tmax':350,'T':300,
         'Pmin':0,'Pmax':2000,'P':0,

         # Model-specific inputs
         'Feed Resistance': 0, # Pa*s/uL
         'Tip-to-Extractor Distance': '_form_|float|Tip-to-Extractor Distance (m)|175e-6',
         'Capillary Radius':'_form_|float|Capillary Radius (m)|3.95e-6',
         'Capillary Length':'_form_|float|Capillary Length (m)|1e-5',
         'Divergence': '_form_|float|Divergence (deg)|30',
         'Boost Voltage':'_form_|float|Boost Voltage (V)|0',
         'CR_m': '_form_|float|CR_m|18.06e-9',
         'CR Limit': '_form_|float|CR_limit|3',
         'P-Scale': '_form_|float|P-Scale|0.023',
         'Droplet Energy Loss': '_form_|float|Droplet Energy Loss (V)|0',

         # Standard deviations for model-specific inputs
         'Capillary Radius_std':'_form_|float|Capillary Radius Stddev (m)|0.25e-6',
         }

    UNITS = {'Isp':'s',#'V*s^2/m',
             'Thrust':'N',
             'Mass Flow':'ug/s',
             'Efficiency':'0 to 1',
             'Electric Current':'A',
             'Number of Active Sites':'unitless',
             'Number of Sites In Ion Mode':'unitless',
             'Median CR':'N/A',
             'V':'V',
             'Divergence Angle':'deg'
             }

    def fields(x):
        metaTxt = ''#'Calculations for the Capillary IL model are as '\
        #'described in Document X.<br><br>'

        # The baseline matrices will be Ns x Nt x Nv x Np
        dims = (x['Ne']*x['Nspe'], x['T'].size, x['V'].size, x['P'].size)

        # Voltage-axis arrays
        V_return = x['V']
        V, = rename(x, 'V', dims=dims, axis='V')
        new_div = x['new_div']

        # Temperature-axis arrays
        T, = rename(x, 'T', dims=dims, axis='T')
        gamma,rho,vis,epR,K = rename(x['Propellants'], 'Surface_Tension',\
        'Density','Dynamic_Viscosity', 'Relative_Permittivity',
        'Conductivity', dims=dims, axis='T')

        # Pressure-axis arrays
        P, = rename(x, 'P', dims=dims, axis='P')

        ### Site axis arrays if _std is defined, otherwise single values
        Ne, Nspe, m_CR, CR_limit, dG, pScale, q_m,\
        D, Vboost, div,\
        m_q, Rfeed, r, L, eDrop_loss = rename(x,\
        'Ne','Nspe','CR_m','CR Limit','dG','P-Scale','q_m',\
        'Tip-to-Extractor Distance','Boost Voltage','Divergence',\
        'm_q','Feed Resistance','Capillary Radius','Capillary Length','Droplet Energy Loss',\
        dims=dims,axis='Site (Maybe)')
        div = np.cos(np.pi/180*div)

        tmsDict,metaTxt = tooManySites(Ne*Nspe,1e4,Model.UNITS.keys(),dims,metaTxt)
        if tmsDict: return tmsDict,metaTxt

#        print("p['gamma'] = %e" % np.mean(gamma))
#        print("p['vis'] = %e" % np.mean(vis))
#        print("p['K'] = %e" % np.mean(K))
#        print("p['epR'] = %e" % np.mean(epR))
#        print("p['rho'] = %e" % np.mean(rho))
#        print("p['q_m'] = %e" % np.mean(q_m))
#        print("p['m_q'] = %e" % np.mean(m_q))
#        print("p['dG'] = %e" % np.mean(dG))

        ### Hydraulic resistance of the feed might've been supplied by other model
        if type(x['Feed Resistance']) == np.ndarray:
            Rfeed = multiDim(x,'Feed Resistance', dims, x['Independent Variable'])

        ### Hydraulic resistance and hyperbolic parameter, eta0
        Rhyd = 8*vis*L/np.pi/r**4
        eta0 = 1/np.sqrt(1+r/D)
        Rhyd += Rfeed
        del vis,Rfeed,L

        ### Onset voltage distribution
        a = 2*D/eta0
        Pfactor = a*np.sqrt((gamma/(r)-np.minimum(0,P)/2)/EP0)
        Vonset = np.arctanh(eta0)*(1-eta0**2)*Pfactor
        Pc = 2*gamma/(r)
        Ec = np.sqrt(2*Pc/EP0)
        CR = K*Ec*(r)**2/Pc/rho/q_m*Rhyd # Mode switch parameter
        del D,Pc,Ec,r

        ### Ion evaporation onset values
        Qmin = gamma*epR*EP0/rho/K/4
        Emin = gamma**0.5*K**(1/6)*EP0**(-2/3)*Qmin**(-1/6)
        G_E_min = np.sqrt(CHARGE_E**3*Emin/4/np.pi/EP0)
        En_l_min = Emin/(epR*(1+PLANCK*K/(EP0*epR*BOLTZ*T)*np.exp((dG-G_E_min)/BOLTZ/T)))
        j_ion_min = K*En_l_min # ion evaporation density at minimum flow
        del G_E_min,En_l_min

        ### Onset field evaporation currents
        Q0 = gamma*EP0/rho/K
        I0 = np.sqrt(EP0*gamma**2/rho)
        rstar_min = 4*gamma/EP0/(Emin**2)
        I_ion_min = np.pi*(rstar_min**2)*j_ion_min
        del Emin,j_ion_min,rstar_min

        ### Preliminary criteria items
        acts = V >= Vonset # active sites
        p0 = 0.5*pScale*EP0*4/a**2/np.arctanh(eta0)**2/(1-eta0**2)**2
        P_Taylor = p0*(V**2-Vonset**2) + np.maximum(0,P)
        del p0

        ### Ion regime
        i = acts & ((CR>CR_limit) & (P<=0)) # ion-regime criteria
        I_ion, Q_ion, I_drop, Q_drop = (np.zeros(dims) for _ in range(4))
        I_ion[i] = I_ion_min[i] + m_CR/CR[i]*(V[i]-Vonset[i])
        Q_ion[i] = I_ion[i]*m_q/rho[i]
        N_ion = np.copy(i)
        del I_ion_min

        ### Cone jet regime
        i = acts & np.logical_not(i)   # cone-jet criteria
        Q_drop[i] = P_Taylor[i]/Rhyd[i]
        i2 = i & (Q_drop<Qmin)
        Q_drop[i2] = Qmin[i2]
        I_drop[i] = I0[i]*(6.2*np.sqrt(Q_drop[i]/Q0[i]/np.sqrt(epR[i]-1))-2)
        Efield, G_E, En_l, rStar = (np.zeros(dims) for _ in range(4))
        Efield[i] = np.sqrt(gamma[i])*(K[i]**(1/6))*(EP0**(-2/3))*(Q_drop[i]**(-1/6))
        G_E[i] = np.sqrt(CHARGE_E**3*Efield[i]/4/np.pi/EP0)
        En_l[i] = Efield[i]/(epR[i]*(1+PLANCK*K[i]/(epR[i]*EP0*BOLTZ*T[i])*
                                np.exp((dG-G_E[i])/BOLTZ/T[i])))
        rStar[i] = 4*gamma[i]/EP0/Efield[i]**2
        I_ion[i] = K[i]*En_l[i]*np.pi*rStar[i]**2
        Q_ion[i] = I_ion[i]*m_q/rho[i]
        del Efield,G_E,En_l,rStar,Vonset,Rhyd,eta0,a,Q0,I0,\
        P_Taylor,Qmin

        CR = np.squeeze(np.median(CR,axis=0))

        ### Key parameters from intermediate data products
        m_q_drop = np.zeros(dims)
        i = I_drop > 0
        m_q_drop[i] = Q_drop[i]*rho[i]/I_drop[i]
        I_El = I_ion + I_drop
        P_El = V*I_El
        mFlow = rho*(Q_ion + Q_drop)
        eta_flow = np.sqrt(K*mFlow/(gamma*epR*EP0))
        div_mult = x['Div Mult']
        #div_angle = np.degrees(np.arctan(div_mult*eta_flow**(3/4)/(K**(1/4)*V**(3/4))))
        div_angle_drop = np.degrees(np.arctan(4.5*np.sqrt(I_drop*np.sqrt(m_q_drop/2)*V**(-3/2)/(2*np.pi*EP0))))
        div_angle = np.degrees(np.arctan(4.5*np.sqrt(I_ion*np.sqrt(m_q/2)*V**(-3/2)/(2*np.pi*EP0)))) + div_angle_drop
        if new_div:
            div = np.cos(np.pi/180*div_angle)
        else:
            div = np.cos(np.pi/180*div)
        div_angle = np.squeeze(np.mean(div_angle,axis=0))
        thrust_drop = I_drop*np.sqrt(2*(V+Vboost-eDrop_loss)*m_q_drop)*div
        thrust = I_ion*np.sqrt(2*(V+Vboost)*m_q)*div+thrust_drop
        del m_q_drop,thrust_drop,rho,gamma,K,epR

        N_ion,P_El,I_El, thrust, mFlow, acts = \
        sumOverNs(N_ion,P_El,I_El, thrust, mFlow, acts)
        Isp,Eff = (np.zeros(thrust.shape) for _ in range(2))
        i = mFlow > 0
        Isp[i] = thrust[i]/mFlow[i]/GRAV
        i = (mFlow > 0) & (np.abs(P_El)>0)
        Eff[i] = thrust[i]**2/2/mFlow[i]/P_El[i]

        return {'Isp':Isp,
                'Thrust':thrust,
                'Mass Flow':mFlow*1e9, # kg/s to ug/s
                'Efficiency':Eff,
                'Electric Current':I_El,
                'Number of Active Sites':acts,
                'Number of Sites In Ion Mode':N_ion,
                'Median CR':CR,
                'Divergence Angle':div_angle,
                'V':V_return}, metaTxt


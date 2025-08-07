import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

norm = np.linalg.norm
# dict_2_printable = lambda k: '\n'.join([f'{keys}: {vals}' for keys,vals in zip(k.keys(), k.values())])

# output printing function
def dict_2_printable(data, indent=0):
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append("  " * indent + f"{key}:")
            lines.append(dict_2_printable(value, indent+1))
        else:
            lines.append("  " * indent + f"{key}: {value}")
    return '\n'.join(lines)


class Gas:
    """Perfect Gas"""
    def __init__(self, gamma, R) -> None:
        self.gamma = gamma
        self.cp = gamma * R / (gamma -1)
        self.cv = self.cp - R
    

# define gas objects
air = Gas(1.40, 287.05)
exhaust = Gas(1.333, 287.05) # placeholder values for hot gas


class V_triangle:
    """ process:
       1) take the inlet velocity
       2) adjust it to the frame of reference of the cascade (be it rotor or stator)
       3) calculate deflection
       4) flow leaves (all outflows are relative to the cascade) """
    def __init__(self, v_inlet: np.ndarray,
                  v_blade: float,
                    turn_angle: float):
        """ v_inlet is the vector velocity of the incoming flow. Type must be array of dimension 2
            v_blade is the scalar speed of the rotor
            turn_angle is the cascade's turning angle in radians"""
        
        # velocity of the blade (positive means right to left motion of the blade)
        # I expect rotors to be positive and stators to be negative

        # type checking
        assert isinstance(v_blade, float)
        self.v_blade = v_blade
        assert isinstance(v_inlet, np.ndarray)
        self.abs_v_inlet = v_inlet
        assert isinstance(v_inlet, np.ndarray), type(v_inlet)
        assert isinstance(turn_angle, float)

        # class input
        self.turn_angle = turn_angle

        # account for the speed of the blade AND the inlet velocity (vector sum)
        self.rel_v_inlet = np.array([v_inlet[0] + v_blade,  
                                     v_inlet[1]])

        # calculate relative exit velocity
        rotation_induced_angle = np.atan2(self.rel_v_inlet[0], self.rel_v_inlet[1]) # angle of attack relative to the blade
        outlet_angle = rotation_induced_angle + turn_angle
        self.v_outlet = np.array([v_inlet[1]*np.tan(outlet_angle),
                                  v_inlet[1]])
 
    def plot(self, title='velocity triangle', verbose=True):
        """ for debugging/viewing
            Note that velocity triangles are drawn upside down in here for convenience."""

        print(f'prior frame of reference v: {np.sqrt(np.dot(self.abs_v_inlet, self.abs_v_inlet))}\n\
        relative inlet velocity: {self.rel_v_inlet}\n\
        outlet velocity: {self.v_outlet}')

        # plot velocity vectors
        # using plt.arrow because plt.quiver is comically broken
        fig, ax = plt.subplots()
        ax.arrow(0,                         0,                     self.abs_v_inlet[0],    self.abs_v_inlet[1],    color='k', head_width=5.0, head_length=5.0)
        ax.arrow(0,                         0,                     self.rel_v_inlet[0],    self.rel_v_inlet[1],    color='r', head_width=5.0, head_length=5.0)
        # ax.arrow(self.abs_v_inlet[0],       self.rel_v_inlet[1],   self.v_blade,           0,                      color='g', head_width=5.0, head_length=5.0)
        # blade velocity = difference from relative inlet to absolute inlet
        blade_vec = self.abs_v_inlet - self.rel_v_inlet
        ax.arrow(self.rel_v_inlet[0], self.rel_v_inlet[1],
                blade_vec[0], blade_vec[1],
                color='g', head_width=5.0, head_length=5.0)

        ax.arrow(0,                         0,                     self.v_outlet[0],       self.v_outlet[1],       color='b', head_width=5.0, head_length=5.0)
        fig.suptitle(f'{title}') 

        # legend (in order)
        plt.legend([f'prior frame of reference v',
                    f'relative inlet v',
                    f'blade velocity (1d)',
                    f'relative outlet velocity'])

        # save graphs or view graphs depending on parameter "verbose"
        if verbose:
            plt.show()
        elif not verbose:
            plt.savefig(f'{title}.png')
            plt.clf()


class TurbineStageStreamline:
    def __init__(self, m_flow, eta_t, T01, p01, pressureRatio, RPM, ht_ratio=0.35, OD=0.4,phi=0.8,alpha3=10):
        self.m_flow = m_flow
        self.eta_t = eta_t
        self.T01 = T01
        self.p01 = p01
        self.p03 = p01 / pressureRatio
        self.RPM = RPM
        self.OD = OD
        self.ht_ratio = ht_ratio
        self.phi = phi
        self.alpha3 = alpha3
        self.gamma = exhaust.gamma
        self.cp = exhaust.cp
        self.R = 287
        self.Ca2 = None
        self.C2 = None
        self.alpha2 = None
        self.beta2 = None
        self.beta3 = None
        self.hub_dia = None
        self.rm = None
        self.U = None
        self.A2 = None
        self.T03 = None
        self.delta_T0s = None
        self.radius_ratios = None
    def run_meanline_design(self):
        
        # power and energy calculations
        self.Power = 1.7e6 # target shaft power output in W
        self.TW = self.Power / self.m_flow # work extracted per unit mass (J/kg)
        self.T03 = self.T01 - self.TW / self.cp # outlet total temp after turbine
        self.delta_T0s = (self.T01 - self.T03) * self.eta_t # isentropic temperature drop

        # rotor speed and geometry
        N = self.RPM / 60 # convert RPM to revolutions per second
        self.hub_dia = self.ht_ratio * self.OD # hub diameter from tip diameter
        self.rm = 0.5 * (self.OD + self.hub_dia) / 2 # mean radius of the blade row
        self.U = N * 2 * np.pi * self.rm # blade speed at mean radius

        # velocity triangle params
        self.psi = 2 * self.cp * self.delta_T0s / self.U**2 # temperature drop coeff
        self.beta3 = np.rad2deg(np.arctan(np.tan(np.deg2rad(self.alpha3)) + 1 / self.phi))  # rotor exit relative flow angle
        self.Lambda = self.phi * np.tan(np.deg2rad(self.beta3)) - self.psi / 4 # degree of reaction
        self.beta2 = np.rad2deg(np.arctan(1 / (2 * self.phi) * (0.5 * self.psi - 2 * self.Lambda))) # rotor inlet relative angle
        self.alpha2 = np.rad2deg(np.arctan(np.tan(np.deg2rad(self.beta2)) + 1 / self.phi)) # rotor inlet absolute angle

        # flow and thermo at station 2
        self.Ca2 = self.U * self.phi # axial velocity at rotor exit
        self.C2 = self.Ca2 / np.cos(np.deg2rad(self.alpha2)) # absolute velocity at rotor exit
        self.T2 = self.T01 - self.C2**2 / (2 * self.cp) # static temp at rotor exit
        self.lambda_N = 0.05 # nozzle loss coeff
        self.T2Prime = self.T2 - self.lambda_N * self.C2**2 / (2 * self.cp) # T2 after nozzle loss ( lambda_N = 0.05)
        self.p2 = self.p01 * (self.T2Prime / self.T01)**(self.gamma / (self.gamma - 1)) # static pressure at station 2
        self.rho2 = self.p2 / (self.R * self.T2) # density at station 2
        self.A2 = self.m_flow / (self.rho2 * self.Ca2) # annulus area needed to pass flow at station 2
        
        # station 1 (inlet) axial velocity
        self.Ca3 = self.Ca2 # axial velocity assumed constant through turbine

        # station 1 velocity components
        self.Ca1 = self.Ca3 / np.cos(np.deg2rad(self.alpha3))
        self.C1 = self.Ca1 # assuming purely axial velocity at inlet (no swirl)
        self.C3 = self.C1 # velocity magnitude at outlet assumed equal to inlet

        # kinetic energy term for inlet/outlet velocity
        self.KE = self.C1**2 / (2 * self.cp)

        # static temp at inlet considering kinetic energy
        self.T1 = self.T01 - self.KE
        self.p1 = self.p01 * (self.T1 / self.T01)**(self.gamma / (self.gamma - 1))
        self.rho1 = self.p1 / (self.R * self.T1)
        self.A1 = self.m_flow / (self.rho1 * self.Ca1)

        # outlet static temp and pressure
        self.T3 = self.T03 - self.KE
        self.p3 = self.p03 * (self.T3 / self.T03)**(self.gamma / (self.gamma - 1))
        self.rho3 = self.p3 / (self.R * self.T3)
        self.A3 = self.m_flow / (self.rho3 * self.Ca3)

        # mean blade speed
        self.Um = self.U

        # annulus area array for stations 1,2,3
        self.A = np.array([self.A1, self.A2, self.A3])

        # blade height estimation (h) at each station
        self.h = self.A * N / self.Um 

        # radius ratio between blade heights
        self.r_ratio = (self.rm + self.h / 2) / (self.rm - self.h / 2)

        # mach number at station 3 (outlet)
        self.M3 = self.C3 / np.sqrt(self.gamma * self.R * self.T3)

    def calc_radial_equilibrium(self):
        """Calculate blade angles at root and tip using free vortex design"""

        # mean radius values
        self.r_m = self.rm
        # blade heights at each station (1 = stator inlet, 2 = rotor inlet, 3 = turbine outlet)
        self.h = self.A * (self.RPM / 60) / self.U

        # root and tip radii at all stations
        self.r_r = [self.r_m - h / 2 for h in self.h] # root radii [r_r1, r_r2, r_r3]
        self.r_t = [self.r_m + h / 2 for h in self.h] # tip radii [r_t1, r_t2, r_t3]

        # free vortex calculations at rotor inlet (station 2)

        self.alpha2_root = np.rad2deg(np.arctan((self.r_m/self.r_r[1]) * np.tan(np.deg2rad(self.alpha2))))
        self.alpha2_tip = np.rad2deg(np.arctan((self.r_m/self.r_t[1]) * np.tan(np.deg2rad(self.alpha2))))

        self.beta2_root = np.rad2deg(np.arctan(np.tan(np.deg2rad(self.alpha2_root)) - (self.r_r[1] / self.r_m) * (1 / self.phi)))
        self.beta2_tip = np.rad2deg(np.arctan(np.tan(np.deg2rad(self.alpha2_tip)) - (self.r_t[1] / self.r_m) * (1 / self.phi)))

        # similarly for outlet angles (station 3)

        self.alpha3_root = np.rad2deg(np.arctan((self.r_m / self.r_r[2]) * np.tan(np.deg2rad(self.alpha3))))
        self.alpha3_tip = np.rad2deg(np.arctan((self.r_m / self.r_t[2]) * np.tan(np.deg2rad(self.alpha3))))

        self.beta3_root = np.rad2deg(np.arctan(np.tan(np.deg2rad(self.alpha3_root)) + (self.r_r[2] / self.r_m) * (1 / self.phi)))
        self.beta3_tip = np.rad2deg(np.arctan(np.tan(np.deg2rad(self.alpha3_tip)) + (self.r_t[2] / self.r_m) * (1 / self.phi)))

        # check reaction at root
        self.Lambda_root = 1 - (np.tan(np.deg2rad(self.alpha2_root)) + np.tan(np.deg2rad(self.alpha3_root))) * self.phi / 2

         # Additional useful calculations
        self.radius_ratios = [rt/rr for rt,rr in zip(self.r_t, self.r_r)]  # [rt/rr]_1, [rt/rr]_2, [rt/rr]_3
        self.annulus_flare_angle = np.rad2deg(np.arctan((self.r_t[2]-self.r_t[0])/(self.h[1]*3)))  # approx flare angle

    def calc_blade_params(self):
        """Calculate blade pitch, chord, and number of blades"""

        # from Ainley-Mathieson correlations
        # Stator (nozzle) blades
        self.deflection_N = self.alpha2 # a1 = 0 to a2
        self.s_c_N = 0.86 # from correlation chart for this deflection

        # Rotor blades
        self.deflection_R = self.beta2 - self.beta3
        self.s_c_R = 0.83 # from correlation chart

        # aspect ratio (h/c) - assume 3 for both
        self.h_N = (self.h[0] + self.h[1]) / 2 # mean stator blade height
        self.h_R = (self.h[1] + self.h[2]) / 2 # mean rotor blade height

        self.c_N = self.h_N / 3 # stator chord length
        self.c_R = self.h_R / 3 # rotor chord

        # pitches
        self.s_N = self.s_c_N * self.c_N
        self.s_R = self.s_c_R * self.c_R

        # number of blades
        self.n_N = int(2 * np.pi * self.r_m / self.s_N) # stator blades
        self.n_R = int(2 * np.pi * self.r_m / self.s_R) # rotor blades

        # ensure prime number for rotor blades to avoid vibration
        if not TurbineStageStreamline.is_prime(self.n_R):
            self.n_R = self.find_next_prime(self.n_R)

    def calc_losses(self):
        """Calculate profile, secondary, and tip clearance losses"""
        # profile losses (simplified)
        self.Y_p_N = 0.024 # from correlation for stator
        self.Y_p_R = 0.032 # from correlation for rotor

        # secondary losses
        self.C_L_N = 2 * (self.s_c_N) * (np.tan(np.deg2rad(self.alpha2))) * np.cos(np.deg2rad(self.alpha2))
        self.C_L_R = 2 * (self.s_c_R) * (np.tan(np.deg2rad(self.beta2)) + np.tan(np.deg2rad(self.beta3))) * np.cos(np.deg2rad((self.beta2 + self.beta3) / 2))

        # tip clearance loss (rotor only)
        k_h = 0.01 # 1% clearance
        B = 0.5 # unshrouded
        self.Y_k = B * k_h * (self.C_L_R / self.s_c_R)**2

        # total losses
        self.Y_N = self.Y_p_N + 0.014 * (self.C_L_N / self.s_c_N)**2 # stator
        self.Y_R = self.Y_p_R + 0.014 * (self.C_L_R / self.s_c_R)**2 + self.Y_k # rotor

        # calculate relative velocities and temperatures
        self.V2_rel = self.Ca2 / np.cos(np.deg2rad(self.beta2))  # Relative velocity at rotor inlet
        self.V3_rel = self.Ca3 / np.cos(np.deg2rad(self.beta3))  # Relative velocity at rotor outlet
        
        # relative total temperatures
        self.T02_rel = self.T01 - (self.C1**2 - self.V2_rel**2)/(2*self.cp)  # Rotor inlet
        self.T03_rel = self.T03 - (self.C3**2 - self.V3_rel**2)/(2*self.cp)  # Rotor outlet
        

        # convert to temperature-based coefficients
        self.lambda_N = self.Y_N / (self.T01 / self.T2)
        self.lambda_R = self.Y_R / (self.T03_rel / self.T3)

        # calculate actual efficiency
        self.eta_actual = 1 / (1 + 0.5*(self.Ca2/self.U)*(
        (self.lambda_R/(np.cos(np.deg2rad(self.beta3))**2)) + 
        (self.T3/self.T2)*(self.lambda_N/(np.cos(np.deg2rad(self.alpha2))**2))
        ) / (np.tan(np.deg2rad(self.beta3)) + np.tan(np.deg2rad(self.alpha2)) - (self.U/self.Ca2)))

    # helper functions
    @staticmethod
    def is_prime(n):
        """Helper function: Check if current number is prime number"""
        if n <= 1:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def find_next_prime(n):
        """Helper function: If current number isn't prime, find next closest prime number"""
        while True:
            n += 1
            if TurbineStageStreamline.is_prime(n):
                return n 

    # getter functions for outputting data

    def get_velocities(self):
        """Returns all velocity components"""
        return {
            "axial velocities" : {
                "nozzle inlet [m/s]" : f"{self.Ca1:.1f}",
                "rotor inlet [m/s]" : f"{self.Ca2:.1f}",
                "rotor outlet [m/s]" : f"{self.Ca3:.1f}"
            },
            "absolute velocities" : {
                "nozzle inlet [m/s]" : f"{self.C1:.1f}",
                "rotor inlet [m/s]" : f"{self.C2:.1f}",
                "rotor outlet [m/s]" : f"{self.C3:.1f}"
            },
            "relative velocities" : {
                "rotor inlet [m/s]" : f"{self.V2_rel:.1f}",
                "rotor outlet [m/s]" : f"{self.V3_rel:.1f}"
            },
            "blade speed [m/s]" : f"{self.U:.1f}",
            "outlet mach number" : f"{self.M3:.2f}"
        }
    
    def get_geometry(self):
        return {
            "hub_diameter" : f"{self.hub_dia:.2f}",
            "mean_radius" : f"{self.rm:.2f}",
            "mean radius [m]" : f"{self.r_m:.4f}",
            "mean stator blade height [m]" : f"{self.h_N:.4f}",
            "mean rotor blade height [m]" : f"{self.h_R:.4f}",
            "station 1": {
                "hub radius [m]" : f"{self.r_r[0]:.4f}",
                "tip radius [m]" : f"{self.r_t[0]:.4f}",
                "radius ratio" : f"{self.radius_ratios[0]:.3f}",
                "annulus_area [m^2]" : f"{self.A1:.2f}",
                "blade height [m]" : f"{self.h[0]:.4f}"

            },
            "station 2" : {
                "hub radius [m]" : f"{self.r_r[1]:.4f}",
                "tip radius [m]" : f"{self.r_t[1]:.4f}",
                "radius ratio" : f"{self.radius_ratios[1]:.3f}",
                "annulus_area [m^2]" : f"{self.A2:.2f}",
                "blade height [m]" : f"{self.h[1]:.4f}"
            },
            "station 3" : {
                "hub radius [m]" : f"{self.r_r[2]:.4f}",
                "tip radius [m]" : f"{self.r_t[2]:.4f}",
                "radius ratio" : f"{self.radius_ratios[2]:.3f}",
                "annulus_area [m^2]" : f"{self.A3:.2f}",
                "blade height [m]" : f"{self.h[2]:.4f}",
            },
            "annulus flare angle [deg]" : f"{self.annulus_flare_angle:.1f}"

        }
    
    def get_thermo(self):
        """Returns thermodynamic properties at all stations"""
        return {
            "Station 1" : {
                "pressure [bar]" : f"{self.p1/1e5:.3f}",
                "temperature [K]" : f"{self.T1:.1f}",
                "density [kg/m^3]" : f"{self.rho1:.3f}",
            },
            "Station 2" : {
                "pressure [bar]" : f"{self.p2/1e5:.3f}",
                "temperature [K]" : f"{self.T2:.1f}",
                "density [kg/m^3]" : f"{self.rho2:.3f}",
            },
            "Station 3" : {
                "pressure [bar]" : f"{self.p3/1e5:.3f}",
                "temperature [K]" : f"{self.T3:.1f}",
                "density [kg/m^3]" : f"{self.rho3:.3f}",
            },
            "drxn" : f"{self.Lambda:.3f}",
            "Root drxn" : f"{self.Lambda_root:.3f}",
            "Tip drxn" : f"{(1 - self.Lambda_root):.3f}",
            "Temperature drop coeff" : f"{self.psi:.3f}",
            "flow coefficient" : f"{self.phi:.3f}"
        }

    def get_blade_params(self):
        """Returns blade parameters"""
        return {
            "geometry" : {
                "stator count" : getattr(self,'n_N','N/A'),
                "rotor count" : getattr(self,'n_R', 'N/A'),
                "stator chord [m]" : f"{getattr(self,'c_N',0):.4f}",
                "rotor chord [m]" : f"{getattr(self,'c_R',0):.4f}"
            },
            "dimensionless parameters" : {
                "stator s/c" : "0.86",
                "rotor s/c" : "0.83",
                "stator h/c" : f"{(self.h[0] + self.h[1])/2/self.c_N:.1f}",
                "rotor h/c" : f"{(getattr(self,'h',[0,0,0])[1] + getattr(self,'h',[0,0,0])[2])/2/getattr(self,'c_R',1):.1f}"
            },
            "Stator angles" : {
                "a1 [deg]" : "0",
                "a2 mean [deg]" : f"{self.alpha2:.2f}",
                "a2 at root [deg]" : f"{self.alpha2_root:.2f}",
                "a2 at tip [deg]" : f"{self.alpha2_tip:.2f}",
                "deflection [deg]" : f"{self.deflection_N:.2f}",
            },
            "Rotor angles" : {
                "b2 mean [deg]" : f"{self.beta2:.2f}",
                "b2 at root [deg]" : f"{self.beta2_root:.2f}",
                "b2 at tip [deg]" : f"{self.beta2_tip:.2f}",
                "b3 mean [deg]" : f"{self.beta3:.2f}",
                "b3 at root [deg]" : f"{self.beta3_root:.2f}",
                "b3 at tip [deg]" : f"{self.beta3_tip:.2f}",
                "a3 mean [deg]" : f"{self.alpha3:.2f}",
                "a3 at root [deg]" : f"{self.alpha3_root:.2f}",
                "a3 at tip [deg]" : f"{self.alpha3_tip:.2f}",
                "deflection [deg]" : f"{self.deflection_R:.2f}"
            },
            
            
            "tip clearance" : "1 % blade height"
        }

    def get_loss_coeffs(self):
        return {
            "profile loss" : {
                "stator Y_p" : f"{self.Y_p_N:.4f}",
                "rotor Y_p" : f"{self.Y_p_R:.4f}",
            },
            "secondary loss" : {
                "stator C_L" : f"{self.C_L_N:.4f}",
                "rotor C_L" : f"{self.C_L_R:.4f}"
            },
            "tip clearance loss (rotor only)" : {
                "rotor Y_k" : f"{self.Y_k:.4f}"
            },
            "total loss" : {
                "stator Y_N" : f"{self.Y_N:.4f}",
                "rotor Y_N" : f"{self.Y_R:.4f}"
            },
            "equivalent losses" : {
                "stator lambda_N" : f"{self.lambda_N:.3f}",
                "rotor lambda_R" : f"{self.lambda_R:.3f}"
            },
            "stage efficiency" : f"{self.eta_actual:.3f}",
            "target efficiency" : f"{self.eta_t:.3f}"

        }
    def plot_v_triangles(self):


        V1_abs = np.array([0.0, self.C1]) # purely axial inlet velocity at station 1

        alpha2_rad = np.deg2rad(self.alpha2)
        V2_abs = np.array([
            self.C2 * np.sin(alpha2_rad), # tangential
            self.C2 * np.cos(alpha2_rad) # axial
        ])

        # stator triangle info
        # New stator triangle (axial inflow (α1=0))
        vstator = V_triangle(
            v_inlet=V1_abs,
            v_blade= 0.0,                        # stator doesn't move
            turn_angle= np.deg2rad(self.alpha2)  # stator turns flow to α2
        )

        # create rotor velocity triangle object
        vtrotor = V_triangle(
            v_inlet=V2_abs,
            v_blade=float(self.U),                             # blade speed at mean radius
            turn_angle=np.deg2rad(self.beta2 - self.beta3)     # cascade turning angle = beta2 - beta3 (in radians)
        )

        # plot results
        vstator.plot(title="IGV Velocity Triangle")
        vtrotor.plot(title="Rotor Velocity Triangle")


def main():
    m_dot = 8 # kg/s
    eta_t = 0.88 # isentropic eff
    T01 = 1173 # K
    p01 = 476300 # Pa
    pressureRatio = 2.16
    RPM = 25650 # from compressor design point

    stage = TurbineStageStreamline(
        m_flow = m_dot,
        eta_t = eta_t,
        T01 = T01,
        p01 = p01,
        pressureRatio=pressureRatio,
        RPM = RPM
    )
    # perform calculations
    stage.run_meanline_design()
    stage.calc_radial_equilibrium()
    stage.calc_blade_params()
    stage.calc_losses()
    print(f'velocities:----------\n'
        f'{dict_2_printable(stage.get_velocities())}\n\n'
        f'geometry:-------------\n'
        f'{dict_2_printable(stage.get_geometry())}\n\n'
        f'thermodynamics:-------\n'
        f'{dict_2_printable(stage.get_thermo())}\n\n'
        f'blade parameters------\n'
        f'{dict_2_printable(stage.get_blade_params())}\n\n'
        f'loss coefficients-----\n'
        f'{dict_2_printable(stage.get_loss_coeffs())}\n\n'
    )
    
    # plot velocity triangles
    stage.plot_v_triangles()

if __name__ == "__main__":
    main()

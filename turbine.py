import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

norm = np.linalg.norm
dict_2_printable = lambda k: '\n'.join([f'{keys}: {vals}' for keys,vals in zip(k.keys(), k.values())])


class Gas:
    """Perfect Gas"""
    def __init__(self, gamma, R) -> None:
        self.gamma = gamma
        self.cp = gamma * R / (gamma -1)
        self.cv = self.cp - R
    

# define gas objects
air = Gas(1.40, 287.05)
exhaust = Gas(1.333, 287.05) # placeholder values for hot gas

class Deflection:

    """Use 1D deflection of a moving gas"""
    def __init__(self, V_inlet) -> None:
        self.V_inlet = np.array(V_inlet, dtype=float)
        self.deflection_angle = None # deg
        self.initial_angle = 0

    def input_deflection_angle(self, deflection_angle):
        """Get user inputted angle"""
        self.deflection_angle = deflection_angle
    
    def calc_outlet_v(self):
        deflection_angle_rad = np.deg2rad(self.deflection_angle)
        
        theta_inlet = np.arctan2(self.V_inlet[1], self.V_inlet[0])
        theta_outlet = theta_inlet - deflection_angle_rad
        speed = norm(self.V_inlet)

        self.V_outlet = np.array([
            speed * np.cos(theta_outlet),
            speed * np.sin(theta_outlet)
        ])
    
    def outlet_V(self):
        assert self.deflection_angle is not None, "you forgor to assign an angle 💀"
        return self.V_outlet


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
        self.h = self.A * (self.RPM / 60) / self.U # Blade heights at stations 1, 2, 3

        # root and tip radii at station 2 (rotor inlet)
        self.r_r = self.r_m - self.h[1] / 2 # root
        self.r_t = self.r_m + self.h[1] / 2 # tip

        # free vortex calculations
        self.alpha2_root = np.rad2deg(np.arctan((self.r_m/self.r_r) * np.tan(np.deg2rad(self.alpha2))))
        self.alpha2_tip = np.rad2deg(np.arctan((self.r_m/self.r_t) * np.tan(np.deg2rad(self.alpha2))))

        self.beta2_root = np.rad2deg(np.arctan(np.tan(np.deg2rad(self.alpha2_root)) - (self.r_r / self.r_m) * (1 / self.phi)))
        self.beta2_tip = np.rad2deg(np.arctan(np.tan(np.deg2rad(self.alpha2_tip)) - (self.r_t / self.r_m) * (1 / self.phi)))

        # similarly for outlet angles
        self.alpha3_root = np.rad2deg(np.arctan((self.r_m / self.r_r) * np.tan(np.deg2rad(self.alpha3))))
        self.alpha3_tip = np.rad2deg(np.arctan((self.r_m / self.r_t) * np.tan(np.deg2rad(self.alpha3))))

        self.beta3_root = np.rad2deg(np.arctan(np.tan(np.deg2rad(self.alpha3_root)) + (self.r_r / self.r_m) * (1 / self.phi)))
        self.beta3_tip = np.rad2deg(np.arctan(np.tan(np.deg2rad(self.alpha3_tip)) + (self.r_t / self.r_m) * (1 / self.phi)))

        # check reaction at root
        self.Lambda_root = 1 - (np.tan(np.deg2rad(self.alpha2_root)) + np.tan(np.deg2rad(self.alpha3_root))) * self.phi / 2

    def calc_blade_params(self):
        """Calculate blade pitch, chord, and number of blades"""

        # from Ainley-Mathieson correlations
        # Stator (nozzle) blades
        deflection_N = self.alpha2 # a1 = 0 to a2
        self.s_c_N = 0.86 # from correlation chart for this deflection

        # Rotor blades
        deflection_R = self.beta2 + self.beta3
        self.s_c_R = 0.83 # from correlation chart

        # aspect ratio (h/c) - assume 3 for both
        h_N = (self.h[0] + self.h[1]) / 2 # mean stator blade height
        h_R = (self.h[1] + self.h[2]) / 2 # mean rotor blade height

        c_N = h_N / 3 # stator chord length
        c_R = h_R / 3 # rotor chord

        # pitches
        s_N = self.s_c_N * c_N
        s_R = self.s_c_R * c_R

        # number of blades
        self.n_N = int(2 * np.pi * self.r_m / s_N) # stator blades
        self.n_R = int(2 * np.pi * self.r_m / s_R) # rotor blades

        # ensure prime number for rotor blades to avoid vibration
        if not self.is_prime(self.n_R):
            self.n_R = self.find_next_prime(self.n_R)

    def calc_losses(self):
        """Calculate profile, secondary, and tip clearance losses"""
        # profile losses (simplified)
        self.Y_p_N = 0.024 # from correlation for stator
        self.Y_p_R = 0.032 # from correlation for rotor

        # secondary losses
        C_L_N = 2 * (self.s_c_N) * (np.tan(np.deg2rad(self.alpha2))) * np.cos(np.deg2rad(self.alpha2))
        C_L_R = 2 * (self.s_c_R) * (np.tan(np.deg2rad(self.beta2)) + np.tan(np.deg2rad(self.beta3))) * np.cos(np.deg2rad((self.beta2 + self.beta3) / 2))

        # tip clearance loss (rotor only)
        k_h = 0.01 # 1% clearance
        B = 0.5 # unshrouded
        self.Y_k = B * k_h * (C_L_R / self.s_c_R)**2

        # total losses
        self.Y_N = self.Y_p_N + 0.014 * (C_L_N / self.s_c_N)**2 # stator
        self.Y_R = self.Y_p_R + 0.014 * (C_L_R / self.s_c_R)**2 + self.Y_k # rotor

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
    def is_prime(self, n):
        """Helper function: Check if current number is prime number"""
        if n <= 1:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def find_next_prime(self, n):
        """Helper function: If current number isn't prime, find next closest prime number"""
        while True:
            n += 1
            if self.is_prime(n):
                return n 

    def get_velocities(self):
        return {
            "Ca2" : f"{self.Ca2:.2f}",
            "C2" : f"{self.C2:.2f}",
            "alpha2" : f"{self.alpha2:.2f}",
            "beta2" : f"{self.beta2:.2f}",
            "beta3" : f"{self.beta3:.2f}"
        }
    
    def get_geometry(self):
        return {
            "hub_diameter" : f"{self.hub_dia:.2f}",
            "mean_radius" : f"{self.rm:.2f}",
            "U" : f"{self.U:.2f}",
            "annulus_area" : f"{self.A2:.2f}"
        }
    
    def get_thermo(self):
        return{
            "T01" : f"{self.T01:.2f}",
            "T03" : f"{self.T03:.2f}",
            "p01" : f"{self.p01:.2f}",
            "p03" : f"{self.p03:.2f}",
            "delta_T0s" : f"{self.delta_T0s:.2f}",
            "DRXN" : f"{self.Lambda:.2f}"
        }
    
    def get_radial_equil(self):
        return{
            "Root_drxn" : f"{self.Lambda_root:.2f}"

        }
    def get_blade_params(self):
        pass
    def get_loss_coeffs(self):
        pass
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
    # perform calculation
    stage.run_meanline_design()
    print(f'velocities:-----------\n'
          f'{dict_2_printable(stage.get_velocities())}\n\n'
          f'geometry:-------------\n'
          f'{dict_2_printable(stage.get_geometry())}\n\n'
          f'Thermodynamics:-------\n'
          f'{dict_2_printable(stage.get_thermo())}\n\n')
    
    # plot velocity triangles
    stage.plot_v_triangles()

if __name__ == "__main__":
    main()

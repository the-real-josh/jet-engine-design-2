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

class Turbine:
    def __init__(self):
        pass


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


if __name__ == "__main__":
    main()

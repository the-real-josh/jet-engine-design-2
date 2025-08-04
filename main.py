import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

norm = np.linalg.norm

# no de haller number because you are accelerating the flow


# gas constants for air
class Gas:
    """Perfect gas"""
    def __init__(self,
                 gamma, 
                    R) -> None:
        self.gamma = gamma
        self.cp = (gamma*R/(gamma-1))
        self.cv = (self.cp-R)
# TODO: represent the air and combustion product mixture as a Gas() object
air = Gas(1.40, 287.05)
exhaust = Gas(np.nan, np.nan)

class Deflection:
    """Use  - 1D deflection of a moving gas"""
    
    def __init__(self, V_inlet) -> None:
        self.V_inlet = V_inlet
        self.deflection_angle = None # PLACEHOLDER

    def calc_deflection_angle(self):
        # calculate the deflection angle based on some principle
        self.deflection_angle = None  # PLACEHOLDER
        assert False, "WARNING: you used a stub function without functionality"

    def asdfasdfasdfasdsdf(self):
        # calculate the deflection angle based on some principle
        unit_v = self.V_inlet / norm(self.V_inlet)
        vertical = np.array([0.0, 1.0], dtype=float)
        self.deflection_angle = np.arccos(np.dot(unit_v, vertical))
        print(f'deflection angle required to straighten flow: {np.rad2deg(self.deflection_angle):.2f}°')
        # YOU WANT THE ROTOR TO STRAIGHTEN THE FLOW IN THE ABSOLUTE NOT THE RELATIVE


    def input_deflection_angle(self, deflection_angle):
        # manually input the deflection angle
        self.deflection_angle = deflection_angle

    def outlet_V(self):
        # calculate hte outlet velocity
        assert self.deflection_angle is not None, "you forgor to assign an angle 💀"
        np.tan(4)
        self.V_outlet = np.array([self.V_inlet[1]*np.tan(self.deflection_angle),
                            self.V_inlet[1]])
        return self.V_outlet



class TurbineStageStreamline:
    def __init__(self, 
                    V_inlet: np.ndarray, # inlet velocity from combustor expansion, m/s
                        RPM: float, # RPM 
                            r: float, # current radius in meters
                                T_inlet: float, # temperature in kelvin
                                    p_inlet: float, # pressure of turbine inlet in pascals
                                        IGV_ang: float, # inlet guide vane angle deflection in radians
                                            rotor_ang: float # rotor angle deflection in radians
                                                ) -> None:
        """How an axial turbine works:
            1)  swirl the air round
            2)  use the swirliness to spin the rotor
            3)  Makes shaft power
        Limiting factor is the material science, not the aerodynamics
        
        """

        omega_blade = (RPM*6.28/60.0)         # speed of rotation in rad/sec
        V_blade = np.array([omega_blade*(r), 0.0], dtype=float) # assumes constant r

        h_inlet = exhaust.cp*T_inlet                    # specific enthalpy for the gas 
        h0_inlet = h_inlet + 0.5*norm(V_blade)**2       # TOTAL specific enthalapy for the gas

        self.T_inlet = T_inlet                  # stage inlet temperature in k
        self.p_inlet = p_inlet                  # stage inlet pressure in pa

        # swirl the air
        d_s = Deflection(V_inlet=V_inlet)
        d_s.input_deflection_angle(IGV_ang)
        IGV_outlet_V = d_s.V_outlet

        # re-calculate temps and energy using energy conservation assumption
        h0_between = h0_inlet - h0_inlet

        # adjust the velocity to be in the frame of the rotor
        rotor_inlet_relative_V = IGV_outlet_V + V_blade
        d_r = Deflection(V_inlet=rotor_inlet_relative_V)
        d_r.input_deflection_angle(np.arccos(np.dot(IGV_outlet_V)))
        outlet_V_relative = d_r
        outlet_V = outlet_V_relative - V_blade

        # euler turbomachinery equation
        self.w = (omega_blade*r*norm(IGV_outlet_V) - omega_blade*r*norm(self.v_1_5_true)) 

        # re-calculate re-calculate temps and energy using adiabatic assumption
        # but knowing that the rotor swiped the velocity energy

        

        pass

    def plot_triangles(self):
        pass

class Turbine:
    def __init__(self) -> None:
        # create radial cross sections
        # do the calcs for each

        
        pass


def main():
    m_dot = 20 # kg/sec
    eta_t = 0.9 # isentropic efficiency
    TIT = 1100 # turbine inlet temperature
    RPM = 26500 # same as compressor!
    # outer diameter is the same as the compressor - 0.4 m diameter, or 0.2 m radius
    lambda_n = 0.05 # nozzle loss coefficient initial firs guess
    Psi = lambda delta_T0s, U: 2*air.cp * delta_T0s / U**2 # temperature drop coefficient
    phi = 0.8 # flow coefficient, guess

    DRXN = lambda n: None # degree of reaction - needt o
    #         self.DRXN = 1 - (norm(v_inlet) / (2*v_blade)) * (np.tan(beta1) - np.tan(beta2))


    

if __name__ == "__main__":
    main()
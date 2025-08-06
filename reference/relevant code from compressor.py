import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# gas constants for air
gamma = 1.4
R = 287.05 # J/kg K
cp = (gamma*R/(gamma-1))
cv = (cp-R)


# helper functions
norm = np.linalg.norm
def pol_to_comps(mag, ang, unit='rad'):
    """
    vector in polar form to 2D vector in components

    inputs:
        mag - magnitude of the vector
        ang - angle of the vector
        unit - rad for radians deg for degrees"""
    if unit=='deg':
        ang = np.deg2rad(ang)

    unit_x = np.array([1.0, 0.0]) # i vector

    rot = np.array([[np.cos(ang), np.sin(ang)],
                   [-np.sin(ang), np.cos(ang)]]) # basic rotation matrix
    return unit_x @ rot


def comps_to_pol(vec, out_unit='rad'):
    """2D vector in components into polar form"""

    # type checking
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 2

    if out_unit == 'deg':
        ang = np.rad2deg(np.atan2(vec[1], vec[0]))
    else:
        ang = np.atan2(vec[1], vec[0])

    mag = np.sqrt(np.dot(vec, vec))

    return mag, ang


def turning_angle(V_in: np.ndarray, # velocity into the stage (radial, axial)
                     C_w: float, # desired exit whirl velocity of the stage
                       V_blade: float # velocity of the blade in m/s
                       ):
    # V_in is the fixed-frame velocity into the rotor
    # C_w is the whirl out of the rotor that is known
    # U is the velocity of the blade
    """gives you turning angle in radians.
    If you can pay the price..."""
    C_out = np.array([C_w, V_in[1]])
    U = np.array([0.0, V_blade])
    turning_angle = np.arccos(np.dot(C_out + U, V_in + U) / (norm(C_out + U)*norm(V_in + U)))
    return -turning_angle


def returning_angle(v_1_5_rel: np.ndarray, v_blade: float):
    """gives you the turning angle that turns the flow back to axial in radians.
     Input: 
        v_1_5_rel - blade-fixed reference frame: rotor outlet velocity
        v_blade - the scalar, linear velocity of the current cross section of rotor blade
     Returns:
        The correct deflection angle in radians
            """
    C_in = v_1_5_rel - np.array([v_blade, 0.0]) # absolute velocity into the stator blade rule
    unit = C_in / norm(C_in)
    unit_vertical = np.array([0, 1], dtype=float)
    ang = np.arccos(np.dot(unit, unit_vertical))
    return -ang


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
        ax.arrow(self.abs_v_inlet[0],       self.rel_v_inlet[1],   self.v_blade,           0,                      color='g', head_width=5.0, head_length=5.0)
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


class Stage_1D:
    """inputs: 
        v_inlet                 vector velocity of inlet, m/s
        rpm                     float compressor's rotational speed in revolutions per minute
        r                       float average radius of compressor stage in meters
        T_inlet                 float stage inlet temperature in k
        p_inlet                 float stage inlet pressure in pa
        rot_defl_ang            float rotor deflection angle in radians
        stat_defl_ang           float stator deflection angle in radian
    
    class values (public):
        T_inlet                 stage inlet temperature in k
        p_inlet                 stage inlet pressure in pa
        stator                  class instance of the stator
        rotor                   class instance of the rotor
        w                       specific work from the compressor
        T_outlet                emperature of stage outlet gas, Kelvin
        isentropic_p_outlet     pressure of the stage outlet in Pa, if the stage was isentropic
        poly_n                  polytropic efficiency (assumed to be 0.9)
        p_outlet                stage outlet pressure based on the polytropic efficiency, in Pa
        phi                     stage flow coefficient
        DRXN                    stage degree of reaction
        DHN                     stage de Haller number
        """
    __total_instance_counter = 0

    # NOTE: please add real imperfect gas behavior as it can be very easy in python    
    def __init__(self,
                  v_inlet: np.ndarray, # meters per second
                    rpm: float, # revolutions per minute
                     r: float, # meter
                       T_inlet:float, # kelvin
                         p_inlet:float, # pascals
                           rot_defl_ang=None, # radians
                             stat_defl_ang=None, # radians
                               K=None,
                                 s_inlet=-1.0):
        
        # ensure that either constant whirl or a set deflection has been defined.
        assert sum([int(bool(K is None)),  int(bool(rot_defl_ang is None and stat_defl_ang is None))]) == 1, f"need one of two to be true: defined K {K} OR defined defl {rot_defl_ang, stat_defl_ang}"

        Stage_1D.__total_instance_counter += 1 # increase the counter of the protected value
        self.instance_number = Stage_1D.__total_instance_counter

        omega_blade = (rpm*6.28/60.0)         # speed of rotation in rad/sec
        v_blade = omega_blade*(r)               # assumes constant r
        h_inlet = cp*T_inlet                    # specific enthalpy for the gas 
        self.T_inlet = T_inlet                  # stage inlet temperature in k
        self.p_inlet = p_inlet                  # stage inlet pressure in pa


        # rotor of the stage
        self.v_inlet = v_inlet

        if rot_defl_ang is None:
            rot_defl_ang = turning_angle(self.v_inlet, K/r, v_blade) # type: ignore
        self.rot_defl_ang = rot_defl_ang
        self.rotor = V_triangle(v_inlet, v_blade, rot_defl_ang)
        v1_5_rel  = self.rotor.v_outlet #  << this is relative to the moving blade!!

        # stator of the stage
        if stat_defl_ang is None:
            stat_defl_ang = returning_angle(v1_5_rel, v_blade) # type: ignore
        self.stat_defl_ang = stat_defl_ang
        self.stator = V_triangle(v1_5_rel, -v_blade, -stat_defl_ang)
        self.v_1_5_true = self.stator.rel_v_inlet # confirm with triangles to make sure this is really true
        self.v2 = self.stator.v_outlet

        # euler's equation for turbomachinery - since the compressor q
        # specific work (energy per mass flow); 
        # NOTE: assuming that this includes all enthalpy added (including velocity)
        # this will be negative, as the compressor REQUIRES energy to operate
        # NOTE: this is only valid for constant velocity along the length of the blade (only radially narrow streamtubes) 
        self.w = (omega_blade*r*norm(v_inlet) - omega_blade*r*norm(self.v_1_5_true)) 

        # outlet enthalpy and temp (static), assumes roughly constant axial velocity
        # negative self.w because the work the work that is put into the system (negative) ADDS to the energy in the system
        # why 0.5*(norm(v_inlet)**2 - norm(self.v2))**2? To account for static change, not stagnation
        h_outlet = -self.w + h_inlet # + 0.5*(norm(v_inlet)**2 - norm(self.v2))**2 

        # get outlet temperature by perfect gas laws
        self.T_outlet = h_outlet/cp

        # get outlet pressures by polytropic gas laws
        # assumes polytropic efficiency = 0.90
        # I don't quite understand the reasoning behind polytropic efficiencies
        self.isentropic_p_outlet = self.p_inlet*(self.T_outlet/T_inlet)**(gamma/(gamma-1))
        self.poly_n = 0.89
        self.p_outlet = self.p_inlet*(self.T_outlet/T_inlet)**((gamma/(gamma-1))*(self.poly_n))

        # flow coefficient
        self.phi = norm(v_inlet)/v_blade

        # worst-case mach number
        a = np.sqrt(gamma*R*T_inlet) # maybe T_inlet is wrong, but it certainly will result in a worst-case M
        M = v1_5_rel / a # v1.5 is probably wrong

        # degree of reaction (called Λ, but python and I hate non-roman variable names)
        # tan(alpha2) - tan(alpha1) = tan(beta1) - tan(beta2)
        #                                                       beta   1                beta 2
        beta1 = np.arctan2(self.rotor.rel_v_inlet[0], self.rotor.rel_v_inlet[1])
        beta2 = stat_defl_ang
        self.DRXN = 1 - (norm(v_inlet) / (2*v_blade)) * (np.tan(beta1) - np.tan(beta2))

        # de haller number
        self.DHN = norm(self.rotor.v_outlet) / norm(self.rotor.rel_v_inlet)

        # recommended outlet area (ratio)
        rho_1 = self.p_inlet  / (R*self.T_inlet)
        rho_2 = self.p_outlet / (R*self.T_outlet)
        self.A_ratio = (rho_1 * norm(v_inlet)) / (rho_2 * norm(self.v2))


        if (s_inlet < 0.0):
            self.s_inlet = cp*np.log(self.T_inlet/273.16) - R*np.log(self.p_inlet/101325.0) + 6608.1
        else:
            self.s_inlet = s_inlet
        self.s_outlet =  cp*np.log(self.T_outlet/self.T_inlet) - R*np.log(self.p_outlet/self.p_inlet) + self.s_inlet        

    def plot_triangles(self, name="", verbose=True):
        """Plots the two velocity triangles for the stage"""
        if name=="":
            name = self._inst_name()
        self.rotor.plot(title=f'Velocity triangle: {name} rotor', verbose=verbose)
        self.stator.plot(title=f'Velocity triangle: {name} stator', verbose=verbose)

    def _inst_name(self):
        # for var_name, var_val in globals().items():
        #     if var_val is self:
        #         return str(var_name)
        return f'stage {self.instance_number}'


    def print_stats(self, verbose=True):
        """ prints out important data for the current stage
            returns a string containing all the stats"""
        status_text = [f'stats for {self._inst_name()}:',
        f'----------------------------',
        f'De Haller number: {self.DHN:.2f}',
        f'Degree of reaction: {self.DRXN:.2f}',
        f'flow coefficient: {self.phi:.2f}',
        f'stage work: {self.w:.1f} J/(kg*sec)',
        f'mass flux: {self.p_inlet/(R*self.T_inlet)*norm(self.v_inlet)}',
        f'stage pressure: {self.p_inlet:.1f} Pa -> {self.p_outlet:.1f} Pa (Ratio of {self.p_outlet/self.p_inlet:.3f})',
        f'temperature (inlet->outlet): {self.T_inlet:.2f} K -> {self.T_outlet:.2f} K  (Ratio of {self.T_outlet/self.T_inlet:.3f})',
        f'entropy (inlet->outlet): {self.s_inlet:.2f}->{self.s_outlet:.2f} Generated {self.s_outlet - self.s_inlet:.2f} J/kg K of entropy',
        f'area ratio: (out/in): {self.A_ratio:.4f}']
        if verbose:
            print('\n'.join(status_text), end='\n\n')
        return status_text

    def plot_mollier(self, verbose=True):
        """ Plots a mollier diagram of the stage's compression process

            verbose:
                If verbose, then the plots are shown and then discarded
                if not verbose, then plots will be written on the global plt object, allowing other methods to annotate after this method."""
        # use this to plot a mollier diagram of the compresion process

        # need to get s as a function of temperature or enthalpy
        # need to do this for isentropic and polytropic processes
        p_polytropic_ratio = lambda T: (T/self.T_inlet)**((gamma/(gamma-1))*(0.9))
        p_isentropic_ratio = lambda T: (T/self.T_inlet)**(gamma/(gamma-1))

        s_isentrope = lambda T: cp*np.log(T/self.T_inlet) - R*np.log(p_isentropic_ratio(T)) + self.s_inlet
        s_polytrope = lambda T: cp*np.log(T/self.T_inlet) - R*np.log(p_polytropic_ratio(T)) + self.s_inlet

        # y axis points (usually temperature or enthalpy)
        T_points = np.linspace(self.T_inlet, self.T_outlet, num=50)

        # plot, label, show
        plt.plot(s_isentrope(T_points), T_points); plt.plot(s_polytrope(T_points), T_points)
        plt.ylabel(f'Temperature (K)'); plt.xlabel(f'Entropy (J/Kg K)'); plt.title(f'Mollier diagram for compression process')
        if verbose:
            plt.legend([f'isentropic', f'polytropic process with n={self.poly_n}'])
            plt.show()

class TrueCompressor:
    def __init__(self) -> None:
        # inlet parameters (same as before)
        comp_rpm = 25650.0
        comp_T_inlet = 288.0
        comp_p_inlet = 101325.0
        comp_v_inlet = np.array([0.0, 250.0])


        meanline_rot_defl_ang = -np.deg2rad(np.array([18, 18, 17, 16, 14]))
        meanline_stat_defl_ang = -np.deg2rad(np.array([36.5, 37, 36.5, 36.5, 35]))


        # rudimentary minline calculations
        self.my_prelim_comp = PrelimCompressor(
            inlet_hub_r = 0.1,
            inlet_tip_r = 0.2,
            comp_rpm = comp_rpm,
            comp_T_inlet = comp_T_inlet,
            comp_p_inlet = comp_p_inlet,
            comp_v_inlet = comp_v_inlet,
            rotor_defl_angles = meanline_rot_defl_ang,
            stator_defl_angles = meanline_stat_defl_ang)
        
        # self.my_prelim_comp.print_stats()
        # self.my_prelim_comp.print_illustrations()
        # self.my_prelim_comp.print_triangles()    
        self.my_prelim_comp.print_mollier_triangles()

        # get the mean radius of each of the stages to generate good streamlines
        self.hub_radii = self.my_prelim_comp.hub_radii
        self.tip_radii = self.my_prelim_comp.tip_radii

        # get the constant K = C_w * R from the minline calcs!!
        meanline_K = self.my_prelim_comp.meanline_K 

        # generate some streamlines
        def streamlines_r(n, rmin=0.1, rmax=0.2):
            """ Get a number of streamlines evenly distributed throughout the span of the blade."""
            return np.convolve(np.linspace(rmin, rmax, num=n+1), np.ones(2), 'valid') / 2

        radii = np.array([streamlines_r(3, rmin=hub_r, rmax=tip_r) for hub_r, tip_r in zip(self.hub_radii, self.tip_radii)]) # generates a (ndarray) list(len=num of stages, ) of lists streamlines 
        
        stages = np.array([[None for i in range(len(radii[0]))] for j in range(len(radii))], dtype=object)

        # re-generate stages with blading
        for s_num in range(len(meanline_rot_defl_ang)): # for every stage


            for r_num in range(len(radii[0])):
                r = radii[s_num, r_num]

                if s_num == 0: # for first stage
                    stages[s_num, r_num] = Stage_1D(v_inlet=comp_v_inlet,
                            rpm=comp_rpm,
                              r=r,
                                T_inlet=comp_T_inlet,
                                  p_inlet=comp_p_inlet,
                                    K=meanline_K[0]) # how to get the whirl velocity between rotor and stator/?

                elif s_num >=1: # for all other stages
                    stages[s_num, r_num] = Stage_1D(v_inlet=stages[s_num-1, r_num].v2,
                                                        rpm=comp_rpm,
                                                            r=r,
                                                                T_inlet=stages[s_num-1, r_num].T_outlet,
                                                                    p_inlet=stages[s_num-1, r_num].p_outlet,
                                                                        K=meanline_K[s_num],
                                                                            s_inlet=stages[s_num-1, r_num].s_outlet)
                else:
                    print(f'mald')
        
        # refined areas calculations
        # necessary?

        # backing out final numbers
        self.stages = stages
        self.radii = radii
        self.rotor_blading = -np.rad2deg(np.array([[stage.rot_defl_ang for stage in stages[j]] for j in range(len(stages))]))
        self.stator_blading = -np.rad2deg(np.array([[stage.stat_defl_ang for stage in stages[j]] for j in range(len(stages))]))
        self.DRXN = np.array([[stage.DRXN for stage in stages[j]] for j in range(len(stages))])

    def plot_all_triangles(self, verbose=False):
        for s_num in range(len(self.stages)):
            for r_num in range(len(self.stages[0])):
                self.stages[s_num, r_num].plot_triangles(name=" ", verbose=verbose)
    # to pair up blading and radii, do np.stack((radii, blading), axis=-1)

    def print_stats(self):
        prelim_results = self.my_prelim_comp.print_stats(verbose=False)
        for s in range(len(self.rotor_blading)):
            print(f'\n\n===  stats for『 stage {s+1} 』  ===')
            print('\n'.join(prelim_results["by_stage"][s][1:]))
            print(f'different radial positions: \n')
            for r in range(len(self.rotor_blading[0])):
                print(f'radial location={self.radii[s, r]:.2f}\n'
                      f'rotor deflection={self.rotor_blading[s, r]:.2f}°\n'
                      f'Stator deflection={self.stator_blading[s, r]:.2f}°\n'
                      F'Degree of Reaction={self.DRXN[s, r]:.2f}\n')
        print(f'\n{prelim_results["overall"]}')


    def export_pandas(self):
        pass # stub


    def make_accurate_illustrations(self):

        # 1D stage illustrations
        fig, ax = plt.subplots()
        for s in range(len(self.hub_radii)):
            # radii = np.concatenate((np.array([self.hub_radii[s]]), np.array(self.radii[s, :]), np.array([self.tip_radii[s]])))
            # radii = np.convolve(radii, np.ones(2), 'valid') / 2
            radii = np.linspace(self.hub_radii[s], self.tip_radii[s], num=len(self.radii[0])+1)
            print(radii)
            for r in range(len(self.radii[0])):
                ax.add_patch(patches.Polygon([
                                             (1.5*s/20,                                     radii[0]),
                                             (1.5*s/20 + self.rotor_blading[s,r]/1500,      radii[0]),
                                             (1.5*s/20 + self.rotor_blading[s,r]/1500,      radii[r+1]),
                                             (1.5*s/20,                                     radii[r+1])]))
                ax.add_patch(patches.Polygon([
                                ((1.5*s+0.5)/20,                                     radii[0]),
                                ((1.5*s+0.5)/20 + self.stator_blading[s,r]/1500,      radii[0]),
                                ((1.5*s+0.5)/20 + self.stator_blading[s,r]/1500,      radii[r+1]),
                                ((1.5*s+0.5)/20,                                     radii[r+1])]))
                ax.add_patch(patches.Polygon([
                                             (1.5*s/20,                                     -radii[0]),
                                             (1.5*s/20 + self.rotor_blading[s,r]/1500,      -radii[0]),
                                             (1.5*s/20 + self.rotor_blading[s,r]/1500,      -radii[r+1]),
                                             (1.5*s/20,                                     -radii[r+1])]))
                ax.add_patch(patches.Polygon([
                                ((1.5*s+0.5)/20,                                     -radii[0]),
                                ((1.5*s+0.5)/20 + self.stator_blading[s,r]/1500,      -radii[0]),
                                ((1.5*s+0.5)/20 + self.stator_blading[s,r]/1500,      -radii[r+1]),
                                ((1.5*s+0.5)/20,                                    -radii[r+1])]))
        m = max(self.tip_radii + [(0.05*(len(self.hub_radii)+1))])
        ax.set_xlim((0, 2*m))
        ax.set_ylim((-m, m))
        plt.title(f'stages')
        plt.xlabel(f'degrees of twist divided by 1500')
        plt.ylabel(f'annulus size (meters)')
        plt.show()
        plt.clf()
        pass # stub - need to get area calculations first

    # TODO:         
    # calculate the updated areas based on the sums of the streamlines


class PrelimCompressor:
    # a compressor designed with only 1D calculations in mind
    # useful to get the preliminary area calculations!!


    def __init__(self,
                 inlet_hub_r : float,
                    inlet_tip_r : float,
                        comp_rpm : float,
                            comp_T_inlet : float,
                                comp_p_inlet : float,
                                    comp_v_inlet : np.ndarray,
                                        rotor_defl_angles : np.ndarray,
                                            stator_defl_angles : np.ndarray):

        # radius ranges from 0.1 m to 0.2 m. keep the external radius constant and increase internal radius as indicated by Stage.A2
        
        # first stage dimensions
        inlet_area = np.pi*(inlet_tip_r**2 - inlet_hub_r**2)

        # do constant outer diameter
        hub_radius_from_area = lambda A: np.sqrt((-A+np.pi*tip_radii[0]**2)/np.pi) # assumes constant exterior
        
        assert len(rotor_defl_angles) == len(stator_defl_angles)

        # setup
        stages = []
        self.meanline_K = []
        hub_radii = [inlet_hub_r]
        tip_radii = [inlet_tip_r]

        # for all stages,
        for s in range(len(rotor_defl_angles)):
            rotor_defl_ang = rotor_defl_angles[0]
            stator_defl_ang = stator_defl_angles[0]

            if s == 0:        # special case - first stage
                stages.append(Stage_1D(v_inlet=comp_v_inlet,
                        rpm=comp_rpm,
                            r=0.5*(hub_radii[-1]+tip_radii[-1]),
                            T_inlet=comp_T_inlet,
                            p_inlet=comp_p_inlet,
                                rot_defl_ang=rotor_defl_angles[0],
                                stat_defl_ang=stator_defl_angles[0]))
                hub_radii.append(hub_radius_from_area(inlet_area*stages[-1].A_ratio))
                tip_radii.append(tip_radii[0]) # constant outer radius

                # rotor_defl_angles = rotor_defl_angles[1:]; stator_defl_angles = stator_defl_angles[1:] # removes first entry of deflection angles so the code works
            else: # for all other stages
                # area calcs for second stage 
                stages.append(
                    Stage_1D(v_inlet=stages[-1].v2,
                        rpm=comp_rpm,
                            r=0.5*(hub_radii[-1]+tip_radii[-1]),
                                T_inlet=stages[-1].T_outlet,
                                    p_inlet=stages[-1].p_outlet,
                                        rot_defl_ang=rotor_defl_ang,
                                            stat_defl_ang=stator_defl_ang,
                                                s_inlet=stages[-1].s_outlet)
                                                )
                hub_radii.append(hub_radius_from_area(np.pi*(tip_radii[-1]**2 - hub_radii[-1]**2)*stages[-1].A_ratio))
                tip_radii.append(tip_radii[0])

            # creating K
            self.meanline_K.append((stages[-1].v_1_5_true[0]) * 0.5*(hub_radii[-1]+tip_radii[-1]))
            print(f'whirl velocities that we are going for at the mean line: {stages[-1].v_1_5_true}')


        # these need to be removed because that the area is always calculated based on the previous stage, generating an extra.
        hub_radii.pop(-1); tip_radii.pop(-1)
        # to do: remove these and put the tip angle on the diagrams.

        self.stages = stages
        self.hub_radii = hub_radii
        self.tip_radii = tip_radii

        print(f'Cw*R = K = '
              f'\nfor the meanline: {self.meanline_K}'
              )

    def print_stats(self, verbose=True):
        # prinout stats and triangles
        by_stage_data = []
        for stage in self.stages:
            by_stage_data.append(stage.print_stats(verbose=verbose))
        
        msg = '\n'.join([f'===  overall compressor stats  ===',
        f'total work: {sum([s.w for s in self.stages]):.2f} J',
        f'pressure ratio: {self.stages[-1].p_outlet / self.stages[0].p_inlet:.2f} ({self.stages[0].p_inlet/1000:.2f} kPa --> {self.stages[-1].p_outlet/1000:.2f} kPa)'])
        if verbose:
            print(msg)

        return {"by_stage": by_stage_data,
                "overall": msg}
    

    def print_triangles(self):
        for stage in self.stages:
            stage.plot_triangles()
    
    def print_mollier_triangles(self):
        # printout mollier diagrams
        for stage in self.stages:
            stage.plot_mollier(verbose=False)
        plt.show()

    def print_illustrations(self):
        # 1D stage illustrations
        fig, ax = plt.subplots()
        for i in range(len(self.hub_radii)):
            ax.add_patch(patches.Rectangle((i/20, self.hub_radii[i]), 1/40, (self.tip_radii[i]-self.hub_radii[i])))
            ax.add_patch(patches.Rectangle((i/20, -(self.hub_radii[i]+self.tip_radii[i]-self.hub_radii[i])), 1/40, (self.tip_radii[i]-self.hub_radii[i])))
        m = max(self.tip_radii + [(0.05*(len(self.hub_radii)+1))])
        ax.set_xlim((0, 2*m))
        ax.set_ylim((-m, m))
        plt.title(f'1D stages (in meters)')
        plt.show()
        plt.clf()



def main():
    c = TrueCompressor()
    c.print_stats()
    c.make_accurate_illustrations()
    c.plot_all_triangles()

if __name__ == "__main__":
    main()



import numpy as np

class Device:

    def __init__(self) -> None:
        self.layers = []

    def addLayer(self, L, er, ur=1):
        '''
        Adds a layer with the given parameters.
        '''
        self.layers.append({'L': L, 'er': er, 'ur': ur})

    def computeSMatrix(self, k_0, k_x, k_y):
        S = getInitialSMatrix()

        for layer in self.layers:
            S_add = computeSMatrixUniform(k_x, k_y, layer['er'], layer['ur'], k_0, layer['L'])
            S = starProduct(S_add, S)

        return S
    
class PeriodicDevice(Device):

    def __init__(self, periods) -> None:
        super().__init__()
        self.periods = periods

    def computeSMatrix(self, k_0, k_x, k_y):
        S_cell = super().computeSMatrix(k_0, k_x, k_y)
        S = getInitialSMatrix()

        for b in list(str(bin(self.periods)).split('b')[1])[::-1]:
            if b:
                S = starProduct(S_cell, S)
            S_cell = starProduct(S_cell, S_cell)

        return S
    
def calculateRT(device, wl, theta, phi, p_TE, p_TM, er_ref, er_trn, ur_ref=1, ur_trn=1, backward=False):
    '''
        Inputs:
            device: a device interface which stores the layers and handles computation of its Smatrix
            wl: vacuum wavelength
            theta, phi: incident angles
            p_TE, p_TM: polarisation amplitudes
            er_ref, er_trn: relative permittivities in the reflection and transmission regions
            ur_ref, ur_trn: relative permeabilities in the reflection and transmission regions
    '''
    if backward:
        n_inc = np.sqrt(er_trn*ur_trn)
    else: 
        n_inc = np.sqrt(er_ref*ur_ref)
    
    k_0, k_x, k_y, c_inc_x, c_inc_y = computeIncidentWave(wl, n_inc, theta, phi, p_TE, p_TM)
    S_global = computeGlobalSMatrix(k_x, k_y, er_ref, ur_ref, er_trn, ur_trn, device.computeSMatrix(k_0, k_x, k_y))
    c_inc = np.array([[c_inc_x], [c_inc_y]])

    if backward:
        c_ref = S_global[3] @ c_inc + 0j
        c_trn = S_global[1] @ c_inc + 0j
        k_z_ref = -np.sqrt(er_trn*ur_trn - k_x**2 - k_y**2 + 0j)
        k_z_trn = np.sqrt(er_ref*ur_ref - k_x**2 - k_y**2 + 0j)

        E_z_ref = -(k_x * c_ref[0] + k_y * c_ref[1]) / k_z_ref
        R = np.abs(c_ref[0])**2 + np.abs(c_ref[1])**2 + np.abs(E_z_ref)**2
        E_z_trn = -(k_x * c_trn[0] + k_y * c_trn[1]) / k_z_trn
        T = (np.abs(c_trn[0])**2 + np.abs(c_trn[1])**2 + np.abs(E_z_trn)**2) * np.real((ur_trn * k_z_trn) / (ur_ref * -k_z_ref))

    else:
        c_ref = S_global[0] @ c_inc + 0j
        c_trn = S_global[2] @ c_inc + 0j
        k_z_ref = -np.emath.sqrt(er_ref*ur_ref - k_x**2 - k_y**2 + 0j)
        k_z_trn = np.emath.sqrt(er_trn*ur_trn - k_x**2 - k_y**2 + 0j)

        E_z_ref = -(k_x * c_ref[0] + k_y * c_ref[1]) / k_z_ref
        R = np.abs(c_ref[0])**2 + np.abs(c_ref[1])**2 + np.abs(E_z_ref)**2
        E_z_trn = -(k_x * c_trn[0] + k_y * c_trn[1]) / k_z_trn
        T = (np.abs(c_trn[0])**2 + np.abs(c_trn[1])**2 + np.abs(E_z_trn)**2) * np.real((ur_ref * k_z_trn) / (ur_trn * -k_z_ref))
       
    return np.asscalar(R), np.asscalar(T)

def getInitialSMatrix():
    S11 = np.zeros(2)
    S12 = np.identity(2)
    S22 = np.zeros(2)
    S21 = np.identity(2)

    S = [S11, S12, S21, S22]

    return S

def starProduct(S_a, S_b):
    '''
        Inputs:
            S_a, S_b: List containing the S-matrices
            [S11, S12, S21, S22]

        Output:
            S_ab: the star product of the two S-matrices
    '''
    I = np.identity(2)
    F = S_a[1] @ np.linalg.inv(I - S_b[0] @ S_a[3])
    D = S_b[2] @ np.linalg.inv(I - S_a[3] @ S_b[0])

    S11 = S_a[0] + F @ S_b[0] @ S_a[2]
    S12 = F @ S_b[1]
    S21 = D @ S_a[2]
    S22 = S_b[3] + D @ S_a[3] @ S_b[1]

    return [S11, S12, S21, S22]

def computeSMatrixUniform(k_x, k_y, er, ur, k_0, L):
    '''
    Inputs:
        parameters required to calculate the S-matrix for a given layer
        k_x, k_y: normalised incident wavevector components
        er: relative permittivity
        ur: relative permeability
        k0: free space wavevector
        L: thickness of the layer

    Output:
        S: the scattering matrix [S11, S12, S21, S22]
    '''
    I = np.identity(2)
    
    k_z_0 = np.emath.sqrt(1-k_x**2-k_y**2)
    Q_0 = np.array([[k_x*k_y, 1 - k_x**2], [k_y**2-1, -k_x*k_y]])
    omega_0_inv = -1j/k_z_0*I
    V_0 = Q_0 @ omega_0_inv

    k_z_i = np.emath.sqrt(ur*er-k_x**2-k_y**2)
    Q_i = 1/ur * np.array([[k_x*k_y, ur*er - k_x**2], [k_y**2-ur*er, -k_x*k_y]])
    omega_i_inv = -1j/k_z_i*I
    V_i = Q_i @ omega_i_inv
    V_i_inv = np.linalg.inv(V_i)
    X = np.array([[np.exp(-1j*k_z_i*k_0*L), 0], [0, np.exp(-1j*k_z_i*k_0*L)]])
    
    A = I + V_i_inv @ V_0
    A_inv = np.linalg.inv(A) 
    B = I - V_i_inv @ V_0
    F = np.linalg.inv(A - X @ B @ A_inv @ X @ B) 
    S11 = F @ (X @ B @ A_inv @ X @ A - B)
    S22 = S11
    S12 = F @ X @ (A - B @ A_inv @ B)
    S21 = S12

    return [S11, S12, S21, S22]

def computeGlobalSMatrix(k_x, k_y, er_ref, ur_ref, er_trn, ur_trn, S_device):
    '''
    Inputs:
        k_x, k_y: normalised incident wavevector components
        er_ref, ur_ref: reflection side material
        er_trn, ur_trn: transmission side material
        S_device: S matrix of multilayered device

    Outputs:
        S_global: the global scattering matrix
    '''
    I = np.conjugate(np.identity(2))

    k_z_0 = np.emath.sqrt(1-k_x**2-k_y**2+0j)
    Q_0 = np.array([[k_x*k_y, 1 - k_x**2], [k_y**2-1, -k_x*k_y]])
    omega_0_inv = -1j/k_z_0*I
    V_0_inv = np.linalg.inv(Q_0 @ omega_0_inv)

    k_z_trn = np.emath.sqrt(ur_trn*er_trn-k_x**2-k_y**2)
    Q_trn = 1/ur_trn * np.array([[k_x*k_y, ur_trn*er_trn - k_x**2], [k_y**2-ur_trn*er_trn, -k_x*k_y]])
    omega_trn_inv = -1j/k_z_trn*I
    V_trn = Q_trn @ omega_trn_inv
    
    k_z_ref = np.emath.sqrt(ur_ref*er_ref-k_x**2-k_y**2)
    Q_ref = 1/ur_ref * np.array([[k_x*k_y, ur_ref*er_ref - k_x**2], [k_y**2-ur_ref*er_ref, -k_x*k_y]])
    omega_ref_inv = -1j/k_z_ref*I
    V_ref = Q_ref @ omega_ref_inv

    A_ref = I + V_0_inv @ V_ref
    B_ref = I - V_0_inv @ V_ref
    A_ref_inv = np.linalg.inv(A_ref)
    S11_ref = -A_ref_inv @ B_ref
    S12_ref = 2*A_ref_inv
    S21_ref = 0.5*(A_ref - B_ref @ A_ref_inv @ B_ref)
    S22_ref = B_ref @ A_ref_inv
    S_ref = [S11_ref, S12_ref, S21_ref, S22_ref]

    A_trn = I + V_0_inv @ V_trn
    B_trn = I - V_0_inv @ V_trn
    A_trn_inv = np.linalg.inv(A_trn)
    S11_trn = B_trn @ A_trn_inv
    S12_trn = 0.5*(A_trn - B_trn @ A_trn_inv @ B_trn)
    S21_trn = 2*A_trn_inv
    S22_trn = -A_trn_inv @ B_trn
    S_trn = [S11_trn, S12_trn, S21_trn, S22_trn]

    return starProduct(S_ref, starProduct(S_device, S_trn))

def computeIncidentWave(wl, n_inc, theta, phi, p_TE, p_TM):
    '''
    Inputs:
        Incident wave parameters
            wl: vacuum wavelength, arbitrary units
            n_inc: incident medium refractive index
            theta: incident angle, in radians
            phi: azimuthal angle, in radians
            p_TE: TE field amplitude
            p_TM: TM field amplitude
    
    Output:
        Incident wavevectors and incident mode coefficients (k_x, k_y, c_inc_x, c_inc_y)
    '''
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    k_x = n_inc * s_theta * c_phi
    k_y = n_inc * s_theta * s_phi
    c_inc_x = -p_TE * s_phi + p_TM * c_theta * c_phi 
    c_inc_y = p_TE * c_phi + p_TM * c_theta * s_phi
    c_inc_z = -p_TM * s_theta

    N = 1/np.sqrt(c_inc_x**2+c_inc_y**2+c_inc_z**2)

    return 2*np.pi / wl, k_x, k_y, c_inc_x/N, c_inc_y/N


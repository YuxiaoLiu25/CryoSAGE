# Adapted from CryoSTAR under Apache-2.0. See NOTICE and provenance.json.
import numpy as np

def euler_angles2matrix(alpha, beta, gamma):
    """
    Converts euler angles in RELION convention to rotation matrix.
    The function is equivalent to "zyz" convention, not the "ZYZ" convention!

    >>> euler_angles2matrix(a, b, c)  
    >>> Rotation.from_euler("zyz", np.array([a, b, c])).as_matrix() # in scipy

    Parameters
    ----------
    alpha: float / np.array
    beta: float / np.array
    gamma: float / np.array

    Returns
    -------
    A: np.array (3, 3)
    """
    # For RELION Euler angle convention
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A = np.zeros((3, 3))
    A[0, 0] = cg * cc - sg * sa
    A[0, 1] = -cg * cs - sg * ca
    A[0, 2] = cg * sb
    A[1, 0] = sg * cc + cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] = sg * sb
    A[2, 0] = -sc
    A[2, 1] = ss
    A[2, 2] = cb
    return A


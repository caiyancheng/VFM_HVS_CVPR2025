import numpy as np

def lms2dkl_d65(lms):
    """
    Convert from LMS color space to DKL color space assuming adaptation to D65 white.

    Parameters:
    lms: ndarray
        The LMS color space values as an array with shape (..., 3).

    Returns:
    dkl: ndarray
        The DKL color space values.
    """

    # The LMS coordinates of the white point (D65 white point)
    lms_gray = np.array([0.739876529525622, 0.320136241543338, 0.020793708751515])

    # Compute constants mc1 and mc2 based on the white point
    mc1 = lms_gray[0] / lms_gray[1]
    mc2 = (lms_gray[0] + lms_gray[1]) / lms_gray[2]

    # Transformation matrix from LMS to DKL
    M_lms_dkl = np.array([[1, 1, 0],
                          [1, -mc1, 0],
                          [-1, -1, mc2]])

    # Perform color space transformation
    dkl = np.dot(lms, M_lms_dkl.T)

    return dkl


def xyz2lms2006(XYZ):
    """
    Transform from CIE 1931 XYZ trichromatic color values to CIE 2006 LMS cone responses.

    Parameters:
    XYZ : ndarray
        The input CIE 1931 XYZ color values. This can be an image (h x w x 3) or a vector (n x 3).

    Returns:
    LMS : ndarray
        The transformed CIE 2006 LMS cone responses.
    """

    # The transformation matrix from CIE 1931 XYZ to CIE 2006 LMS
    M_xyz_lms2006 = np.array([
        [0.187596268556126, 0.585168649077728, -0.026384263306304],
        [-0.133397430663221, 0.405505777260049, 0.034502127690364],
        [0.000244379021663, -0.000542995890619, 0.019406849066323]
    ])

    # Perform the color space transformation
    LMS = np.dot(XYZ, M_xyz_lms2006.T)  # Use matrix multiplication (XYZ * M)

    return LMS

def dkl2lms_d65(dkl):
    """
    Convert from DKL color space to LMS color space assuming adaptation to D65 white.

    Parameters:
    dkl : ndarray
        The DKL color space values (n x 3 or h x w x 3).

    Returns:
    lms : ndarray
        The transformed LMS color space values.
    """

    # The LMS coordinates of the D65 white point
    lms_gray = np.array([0.739876529525622, 0.320136241543338, 0.020793708751515])

    # Calculate the constants mc1 and mc2
    mc1 = lms_gray[0] / lms_gray[1]
    mc2 = (lms_gray[0] + lms_gray[1]) / lms_gray[2]

    # Transformation matrix from LMS to DKL
    M_lms_dkl = np.array([[1, 1, 0],
                          [1, -mc1, 0],
                          [-1, -1, mc2]])

    # Inverse transformation matrix from DKL to LMS
    M_dkl_lms = np.linalg.inv(M_lms_dkl)

    # Perform the color space transformation
    lms = np.dot(dkl, M_dkl_lms.T)

    return lms
def lms2006_2xyz(LMS):
    """
    Transform from CIE 2006 LMS cone responses to CIE 1931 XYZ trichromatic color values.

    Parameters:
    LMS : ndarray
        The CIE 2006 LMS color space values. Can be an image (h x w x 3) or color vectors (n x 3).

    Returns:
    XYZ : ndarray
        The transformed CIE 1931 XYZ color space values.
    """

    # The transformation matrix from CIE 2006 LMS to CIE 1931 XYZ
    M_lms2006_xyz = np.array([
        [2.629129278399650, -3.780202391780134, 10.294956387893450],
        [0.865649062438827, 1.215555811642301, -0.984175688105352],
        [-0.008886561474676, 0.081612628990755, 51.371024830897888]
    ])

    # Perform the color space transformation
    XYZ = np.dot(LMS, M_lms2006_xyz.T)

    return XYZ


def cm_xyz2rgb(xyz, rgb_space='rec709'):
    """
    Convert XYZ color space values to RGB color space values.

    Parameters:
    xyz : ndarray
        The input XYZ color space values. Can be an image (h x w x 3) or a vector (n x 3).
    rgb_space : str, optional
        The RGB color space to convert to. Can be 'Adobe', 'NTSC', 'sRGB', 'rec709', 'rec2020'.
        Default is 'rec709'.

    Returns:
    rgb : ndarray
        The transformed RGB color space values.
    """

    if rgb_space == 'Adobe':
        M_xyz2rgb = np.array([
            [2.04148, -0.969258, 0.0134455],
            [-0.564977, 1.87599, -0.118373],
            [-0.344713, 0.0415557, 1.01527]
        ])
    elif rgb_space == 'NTSC':
        M_xyz2rgb = np.array([
            [1.9099961, -0.5324542, -0.2882091],
            [-0.9846663, 1.9991710, -0.0283082],
            [0.0583056, -0.1183781, 0.8975535]
        ])
    elif rgb_space in {'sRGB', 'rec709'}:
        M_xyz2rgb = np.array([
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570]
        ])
    elif rgb_space == 'rec2020':
        M_xyz2rgb = np.linalg.inv(np.array([
            [0.636953507, 0.144619185, 0.168855854],
            [0.262698339, 0.678008766, 0.0592928953],
            [4.99407097e-17, 0.0280731358, 1.06082723]
        ]))
    else:
        raise ValueError('Unknown RGB color space')

    # Perform the color space transformation
    rgb = np.dot(xyz, M_xyz2rgb.T)

    return rgb

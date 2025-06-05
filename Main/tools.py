import math
import numpy as np

def getError(p, pd, R, persective_flag = True):
    # Vectors should be represented from inertial frame
    # If the perspective flag is true then it returns the error vector from the robots perspective
    e = pd - p

    if persective_flag:
        e = changePerspective(e, R.T)

    return e


def getLinearError(e):
    # Vectors should be represented from robots frame
    return e[0]

def getAngularError(e):
    # Vectors should be represented from robots frame
    unit_x = np.eye(3)[0, :]
    dot_product = unit_x @ make_unit(e)

    cross_product = np.cross(unit_x, e)

    n = cross_product[2] / np.abs(cross_product[2])
    
    angle = n * math.acos(dot_product)
    
    return angle

def make_unit(p):
    return p / np.linalg.norm(p)

def changePerspective(p, R):
    p = R @ p

    return p

def getFifthOrder(t, tf, qi, qdi, qddi, qf, qdf, qddf):
    k0 = qi
    k1 = qdi
    k2 = qddi / 2
    k3 = (10 * (qf-qi) / (tf**3))   - ((4*qdf + 6*qdi) / (tf**2))     - ((3*qddi-qddf) / (2*tf))
    k4 = (-15 * (qf-qi) / (tf**4))  + ((7*qdf + 8*qdi) / (tf**3))   + ((3*qddi-2*qddf) / 2*(tf**2))
    k5 = (6*(qf-qi) / (tf**5))      - ((3*qdf + qdi) / (tf**4))     - ((qddi - qddf) / 2*(tf**3))


    pt = k0 + k1*t + k2*(t**2) + k3*(t**3) + k4*(t**4) + k5*(t**5)
    vt = k1 + 2*k2*t + 3*k3*(t**2) + 4*k4*(t**3) + 5*k5*(t**4)
    at = 2*k2 + 6*k3*t + 12*k4*(t**2) + 20*k5*(t**3)

    if t > tf:
        pt = qf
        vt = qdf
        at = qddf
    
    return pt, vt, at
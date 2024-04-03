# Defines the gauge transformation operator at a given time for every x
@myjit
def trans_operator(t, aeta, ux, uprev): #TODO: How can I automatize it for every x
    buffer1 = su.mexp(1j*g*L/N*z* aeta[t, x, y, :]) * ux[t, x, y, :]
    r = su.mul(buffer1, uprev[(t+x)//2])
    
    return r

#Transforms the field at a given xplus
@myjit
def transform_link(u, v, t):
    u_plus_LC = su.dagger(v[t+1, x+1, y, z]) * u[t, x, y] * v[t,x,y,z]
    
    return u_plus_LC
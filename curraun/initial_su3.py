from curraun.numba_target import myjit
import curraun.su as su

ACCURACY_GOAL = 1e-16 # 1e-8
ITERATION_MAX_ROUND_1 = 250 # 100
ITERATION_MAX_ROUND_2 = 100000 # 10000
HEAVY_BALL_BETA = 1
TRY_FACTORS = [1, 2, 0, -1]

# Gradient descent on algebra element
# Calculate gradient analytically
@myjit
def init_kernel_2_su3(xi, u0, u1, ua, ub):
    # if xi != 594 and xi != 750 and xi != 814:
    #     return
    # initialize transverse gauge links (longitudinal magnetic field)
    # (see PhD thesis eq.(2.135))  # TODO: add proper link or reference
    for d in range(2):
        u_a = su.load(ua[xi, d])
        u_b = su.load(ub[xi, d])

        b3 = solve_initial_condition_complete(u_a, u_b, xi, d)

        su.store(u0[xi, d], b3)
        su.store(u1[xi, d], b3)

# Try different
@myjit
def solve_initial_condition_complete(u_a, u_b, xi, d):
    # Try starting from various initial conditions and see which gets closest to the result.
    best_loss = 1000
    best_factor = 0
    for factor in TRY_FACTORS:
        b3, loss, accuracy_reached, iterations = solve_initial_condition(u_a, u_b, xi, d, factor, ITERATION_MAX_ROUND_1)
        if accuracy_reached:
            # We are done :)
            return b3
        if loss < best_loss:
            best_loss = loss
            best_factor = factor

    # Start from that initial condition and dig really deep
    b3, loss, accuracy_reached, iterations = solve_initial_condition(u_a, u_b, xi, d, best_factor, ITERATION_MAX_ROUND_2)

    if accuracy_reached:
        print("Kernel 2: xi:", xi, ", d:", d, ": digging deep successful:", iterations,
              ", factor: ", factor, ". Loss:", loss)
        return b3

    print("=========================================================================")
    print("WARNING")
    print("Kernel 2: xi:", xi, ", d:", d, ": digging deep unsuccessful. ", iterations,
              ", factor: ", factor, "Loss:", loss)
    print("=========================================================================")
    return b3

@myjit
def solve_initial_condition(u_a, u_b, xi, d, initial_factor, iter_max):
    b1 = su.add(u_a, u_b)  # A
    # Better starting value:
    m1a = su.get_algebra_factors_from_group_element_approximate(u_a)
    m1b = su.get_algebra_factors_from_group_element_approximate(u_b)
    m1 = su.add_algebra(m1a, m1b)
    m1 = su.mul_algebra(m1, initial_factor)
    m1_prev = m1

    epsilon2 = 0.5  # 0.125 # 0.0001 # 0.125
    si3 = 0
    si4 = 0
    smallestloss = 1
    accuracy_reached = False
    for i in range(iter_max):
        # Calculate Loss:
        loss1 = loss(b1, m1)

        m2new = m1

        # Calculate analytic derivative:
        grad = gradient(b1, m1)

        m2new = su.add_algebra(m2new, su.mul_algebra(grad, -epsilon2))

        # Heavy ball method
        # dball = + beta * (m1 - m1_prev)
        dball = su.add_algebra(m1, su.mul_algebra(m1_prev, -1))
        dball = su.mul_algebra(dball, HEAVY_BALL_BETA)

        m2new21 = su.add_algebra(m2new, dball)

        loss3 = loss(b1, m2new)
        loss21 = loss(b1, m2new21)

        m1_prev = m1

        # Find step with smallest value of loss
        smallestloss_prev = smallestloss
        smallestloss = loss1
        smallestitem = -1
        if loss3 < smallestloss:
            m1 = m2new
            smallestloss = loss3
            smallestitem = 5
        if loss21 < smallestloss:
            m1 = m2new21
            smallestloss = loss21
            smallestitem = 21

        if smallestitem == 5:
            si3 += 1
        if smallestitem == 21:
            si4 += 1

        if smallestitem == -1:
            pass

        if smallestloss < ACCURACY_GOAL:
            #    print("Kernel 2: xi:", xi, ", d:", d, ": Iterations:", i, ". Bounds:", loss3, ", eps: ", epsilon2, (si3, si4))
            accuracy_reached = True
            break
    else:  # no break
        # print("Kernel 2: xi:", xi, ", d:", d, ": max iterations reached:", i, ". Bounds:", loss3, ", factor: ", initial_factor,
        #       (si3, si4))
        pass

    b3 = su.mexp(su.get_algebra_element(m1))
    final_loss = loss(b1, m1)
    return b3, final_loss, accuracy_reached, i

@myjit
def loss(b1, m1):
    unit = su.unit()

    # Check result
    b3 = su.mexp(su.get_algebra_element(m1))
    e1 = su.mul(b1, su.dagger(su.add(unit, b3)))
    e2 = su.ah(e1)
    res = su.sq(e2)

    return res

# calculate gradient for loss
@myjit
def gradient(b1, m1):
    grad = su.zero_algebra
    for a in range(8):
        mdelta = su.unit_algebra[a]

        # Calculate analytic derivative in direction mdelta
        grad_a = gradient_component(b1, m1, mdelta)
        grad = su.add_algebra(grad, su.mul_algebra(mdelta, grad_a))
    return grad

@myjit
def gradient_component(b1, m1, mdelta):

    # b1 # A
    unit = su.unit()
    b3 = su.mexp(su.get_algebra_element(m1)) # B = exp(m1)

    e1 = su.mul(b1, su.dagger(su.add(unit, b3))) # X = A (1 + B)^dagger

    e2 = su.add(e1, su.mul_s(su.dagger(e1), -1)) # X - X^dagger
    e3 = su.mul_s(unit, -su.tr(e2) / su.N_C) # - tr(e2) / N_C * 1
    e4 = su.add(e2, e3) # Y = [X]_ah

    m3 = su.get_algebra_element(mdelta)

    db3 = su.dmexp(su.get_algebra_element(m1), su.mul_s(m3, -1)) # derivative of b3
    f1 = su.mul(b1, su.dagger(db3))  # A dB^dagger

    f2 = su.add(f1, su.mul_s(su.dagger(f1), -1)) # f1 - f1^dagger
    f3 = su.mul_s(unit, -su.tr(f2) / su.N_C) # - tr(f2) / N_C * 1
    f4 = su.add(f2, f3) # d/da Y = d/da [X]_ah

    g1 = su.mul(f4, su.dagger(e4)) # (d/da Y) . Y^dagger
    g2 = su.mul(e4, su.dagger(f4)) # Y . d/da Y^dagger
    g3 = su.tr(su.add(g1, g2)) # d/da tr(Y . Y^dagger) = tr(g1 + g2)

    res = -g3.real * 0.25 # Result should be real
    return res

using LinearAlgebra: Matrix
using ControlSystems
import ControlSystems: dare, kalman
"""
Functionality for observing a state (mainly via a Kalman filter), in interaction with a ControlSystems.jl system.
"""
struct KFilter
    A::Matrix # state dynamics
    B::Matrix # input dynamics
    C::Matrix # measurement
    Q::Matrix # process covariance
    R::Matrix # measurement covariance
    K::Matrix # Kalman gain

    function KFilter(A, B, C, Q, R)
        iters = 0
        P = copy(Q)
        lastP = zeros(size(A))
        K = zeros(size(A, 1), size(C, 1))
        while !all(lastP .≈ P)
            lastP = copy(P)
            P = A * P * A' + Q
            K = P * C' * inv(C * P * C' + R)
            P = P - K * C * P
            iters += 1
        end
        println("Took $iters iterations to reach steady-state covariance.")

        new(A, B, C, Q, R, K)
    end
end

predict(kf::KFilter, x::Vector, u::Vector) = kf.A * x + kf.B * u
update(kf::KFilter, x::Vector, y::Vector) = x + kf.K * (y - kf.C * x)

dare(kf::KFilter) = dare(kf.A', kf.C', kf.Q, kf.R)

function kalman(kf::KFilter)
    P = dare(kf)
    return P * kf.C' * inv(kf.C * P * kf.C' + kf.R)
end

function simulate(kf::KFilter, measurements::Vector, inputs::Vector, x0::Vector)
    states = [zeros(size(kf.A, 1)) for _ in measurements]
    x = copy(x0)
    for (i, (u, m)) in enumerate(zip(inputs, measurements))
        x = update(kf, predict(kf, x, u), m)
        states[i] = x
    end
    return states
end

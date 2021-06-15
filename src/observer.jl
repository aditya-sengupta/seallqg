using Core: Matrix
"""
Functionality for observing a state (mainly via a Kalman filter), in interaction with a ControlSystems.jl system.
"""

struct KFilter
    x::Vector # state
    P::Matrix # state covariance
    A::Matrix # state dynamics
    B::Matrix # input dynamics
    C::Matrix # measurement
    Q::Matrix # process noise covariance
    R::Matrix # measurement noise covariance
end

function predict!(kf::KFilter, input::Vector)
    kf.x = kf.A * kf.x + kf.B * input
    kf.P = kf.A * kf.P * kf.A' + kf.Q
end

function update!(kf::KFilter, measurement::Vector)
    
end

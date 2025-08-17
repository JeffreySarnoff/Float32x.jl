"""
    Float32xModule

A module implementing an extended precision floating-point type using a Float32 significand
and Int32 exponent, providing operations similar to Float64.
"""
module Float32xModule

export Float32x, zero, one, inf, nan, eps
export iszero, isone, isinf, isnan, isfinite, issubnormal, signbit
export abs, sign, copysign, flipsign, nextfloat, prevfloat
export round, floor, ceil, trunc, significand, exponent
export sqrt, cbrt, exp, exp2, exp10, expm1
export log, log2, log10, log1p
export sin, cos, tan, asin, acos, atan
export sinh, cosh, tanh, asinh, acosh, atanh
export ldexp, frexp, modf

import Base: +, -, *, /, ^, %, ÷
import Base: ==, !=, <, <=, >, >=, isless, isequal
import Base: zero, one, abs, sign, copysign, flipsign
import Base: iszero, isone, isinf, isnan, isfinite, issubnormal, signbit
import Base: round, floor, ceil, trunc
import Base: sqrt, cbrt, exp, exp2, exp10, expm1
import Base: log, log2, log10, log1p
import Base: sin, cos, tan, asin, acos, atan
import Base: sinh, cosh, tanh, asinh, acosh, atanh
import Base: ldexp, frexp, modf
import Base: show, string, print
import Base: promote_rule, convert
import Base: nextfloat, prevfloat
import Base: significand, exponent
import Base: eps, typemin, typemax
import Base: float, Float64, Float32, Float16
import Base: hash

# ==================== Type Definition ====================

"""
    Float32x <: AbstractFloat

Extended precision floating-point type using Float32 significand and Int32 exponent.
Represents: significand × 2^exponent
"""
struct Float32x <: AbstractFloat
    significand::Float32
    exponent::Int32
    
    # Inner constructor with normalization
    function Float32x(s::Float32, e::Int32)
        # Handle special cases
        if isnan(s)
            return new(NaN32, Int32(0))
        elseif isinf(s)
            return new(s, Int32(0))
        elseif iszero(s)
            return new(0.0f0, Int32(0))
        end
        
        # Normalize: ensure 1.0 <= |significand| < 2.0 for non-zero values
        sig, exp_adj = frexp(s)
        new(Float32(sig * 2), e + Int32(exp_adj) - 1)
    end
end

# Convenience constructor
Float32x(x::Real) = convert(Float32x, x)

# ==================== Constants ====================

Base.zero(::Type{Float32x}) = Float32x(0.0f0, Int32(0))
Base.one(::Type{Float32x}) = Float32x(1.0f0, Int32(0))
inf(::Type{Float32x}) = Float32x(Inf32, Int32(0))
nan(::Type{Float32x}) = Float32x(NaN32, Int32(0))
Base.eps(::Type{Float32x}) = Float32x(eps(Float32), Int32(0))
Base.typemin(::Type{Float32x}) = Float32x(-Inf32, Int32(0))
Base.typemax(::Type{Float32x}) = Float32x(Inf32, Int32(0))

# ==================== Predicates ====================

Base.iszero(x::Float32x) = iszero(x.significand)
Base.isone(x::Float32x) = x.significand == 1.0f0 && x.exponent == 0
Base.isinf(x::Float32x) = isinf(x.significand)
Base.isnan(x::Float32x) = isnan(x.significand)
Base.isfinite(x::Float32x) = isfinite(x.significand)
Base.issubnormal(x::Float32x) = issubnormal(x.significand)
Base.signbit(x::Float32x) = signbit(x.significand)

# ==================== Comparison Operations ====================

Base.:(==)(x::Float32x, y::Float32x) = begin
    if isnan(x) || isnan(y)
        return false
    end
    if iszero(x) && iszero(y)
        return true
    end
    return x.significand == y.significand && x.exponent == y.exponent
end

Base.:(!=)(x::Float32x, y::Float32x) = !(x == y)

Base.isless(x::Float32x, y::Float32x) = begin
    if isnan(x) || isnan(y)
        return false
    end
    if isinf(x) || isinf(y)
        return x.significand < y.significand
    end
    if signbit(x) != signbit(y)
        return signbit(x)
    end
    if x.exponent != y.exponent
        return signbit(x) ? x.exponent > y.exponent : x.exponent < y.exponent
    end
    return x.significand < y.significand
end

Base.:(<=)(x::Float32x, y::Float32x) = x == y || isless(x, y)
Base.:(<)(x::Float32x, y::Float32x) = isless(x, y)
Base.:(>=)(x::Float32x, y::Float32x) = y <= x
Base.:(>)(x::Float32x, y::Float32x) = y < x

Base.isequal(x::Float32x, y::Float32x) = 
    isequal(x.significand, y.significand) && isequal(x.exponent, y.exponent)

# ==================== Conversions ====================

Base.convert(::Type{Float32x}, x::Float32x) = x
Base.convert(::Type{Float32x}, x::Integer) = Float32x(Float32(x), Int32(0))
Base.convert(::Type{Float32x}, x::Float32) = Float32x(x, Int32(0))
Base.convert(::Type{Float32x}, x::Float64) = begin
    if isnan(x) || isinf(x) || iszero(x)
        return Float32x(Float32(x), Int32(0))
    end
    sig, exp = frexp(x)
    Float32x(Float32(sig * 2), Int32(exp - 1))
end

Base.convert(::Type{Float64}, x::Float32x) = 
    Float64(x.significand) * exp2(Float64(x.exponent))

Base.convert(::Type{Float32}, x::Float32x) = 
    x.significand * exp2(Float32(min(max(x.exponent, -126), 127)))

Base.Float64(x::Float32x) = convert(Float64, x)
Base.Float32(x::Float32x) = convert(Float32, x)
Base.float(x::Float32x) = x

Base.promote_rule(::Type{Float32x}, ::Type{<:Integer}) = Float32x
Base.promote_rule(::Type{Float32x}, ::Type{Float32}) = Float32x
Base.promote_rule(::Type{Float32x}, ::Type{Float64}) = Float32x

# ==================== Basic Arithmetic ====================

Base.:(+)(x::Float32x, y::Float32x) = begin
    if isnan(x) || isnan(y)
        return nan(Float32x)
    end
    if isinf(x) || isinf(y)
        if isinf(x) && isinf(y) && signbit(x) != signbit(y)
            return nan(Float32x)
        end
        return isinf(x) ? x : y
    end
    if iszero(x)
        return y
    end
    if iszero(y)
        return x
    end
    
    # Align exponents
    if x.exponent > y.exponent
        diff = x.exponent - y.exponent
        if diff > 24  # Beyond Float32 precision
            return x
        end
        aligned_y = ldexp(y.significand, -diff)
        result = x.significand + aligned_y
    elseif y.exponent > x.exponent
        diff = y.exponent - x.exponent
        if diff > 24
            return y
        end
        aligned_x = ldexp(x.significand, -diff)
        result = aligned_x + y.significand
        return Float32x(result, y.exponent)
    else
        result = x.significand + y.significand
    end
    
    Float32x(result, x.exponent)
end

Base.:(-)(x::Float32x) = Float32x(-x.significand, x.exponent)
Base.:(-)(x::Float32x, y::Float32x) = x + (-y)

Base.:(*)(x::Float32x, y::Float32x) = begin
    if isnan(x) || isnan(y)
        return nan(Float32x)
    end
    if isinf(x) || isinf(y)
        if iszero(x) || iszero(y)
            return nan(Float32x)
        end
        s = sign(x.significand) * sign(y.significand)
        return Float32x(s * Inf32, Int32(0))
    end
    if iszero(x) || iszero(y)
        return zero(Float32x)
    end
    
    Float32x(x.significand * y.significand, x.exponent + y.exponent)
end

Base.:(/)(x::Float32x, y::Float32x) = begin
    if isnan(x) || isnan(y)
        return nan(Float32x)
    end
    if iszero(y)
        if iszero(x)
            return nan(Float32x)
        end
        return Float32x(sign(x.significand) * sign(y.significand) * Inf32, Int32(0))
    end
    if isinf(y)
        if isinf(x)
            return nan(Float32x)
        end
        return zero(Float32x)
    end
    if isinf(x)
        s = sign(x.significand) * sign(y.significand)
        return Float32x(s * Inf32, Int32(0))
    end
    if iszero(x)
        return zero(Float32x)
    end
    
    Float32x(x.significand / y.significand, x.exponent - y.exponent)
end

Base.:(^)(x::Float32x, y::Float32x) = exp(y * log(x))
Base.:(^)(x::Float32x, n::Integer) = begin
    if n == 0
        return one(Float32x)
    elseif n == 1
        return x
    elseif n == 2
        return x * x
    elseif n < 0
        return one(Float32x) / (x^(-n))
    else
        # Binary exponentiation
        result = one(Float32x)
        base = x
        while n > 0
            if isodd(n)
                result *= base
            end
            base *= base
            n >>= 1
        end
        return result
    end
end

# ==================== Pre-arithmetic Operations ====================

Base.abs(x::Float32x) = Float32x(abs(x.significand), x.exponent)
Base.sign(x::Float32x) = Float32x(sign(x.significand), Int32(0))
Base.copysign(x::Float32x, y::Float32x) = 
    Float32x(copysign(x.significand, y.significand), x.exponent)
Base.flipsign(x::Float32x, y::Float32x) = 
    Float32x(flipsign(x.significand, y.significand), x.exponent)

Base.nextfloat(x::Float32x) = begin
    if isnan(x)
        return x
    end
    if isinf(x) && x.significand > 0
        return x
    end
    Float32x(nextfloat(x.significand), x.exponent)
end

Base.prevfloat(x::Float32x) = begin
    if isnan(x)
        return x
    end
    if isinf(x) && x.significand < 0
        return x
    end
    Float32x(prevfloat(x.significand), x.exponent)
end

# ==================== Rounding Operations ====================

Base.round(x::Float32x) = convert(Float32x, round(Float64(x)))
Base.floor(x::Float32x) = convert(Float32x, floor(Float64(x)))
Base.ceil(x::Float32x) = convert(Float32x, ceil(Float64(x)))
Base.trunc(x::Float32x) = convert(Float32x, trunc(Float64(x)))

Base.significand(x::Float32x) = Float64(x.significand)
Base.exponent(x::Float32x) = x.exponent

# ==================== Mathematical Functions ====================

# Exponential functions
Base.exp(x::Float32x) = convert(Float32x, exp(Float64(x)))
Base.exp2(x::Float32x) = convert(Float32x, exp2(Float64(x)))
Base.exp10(x::Float32x) = convert(Float32x, exp10(Float64(x)))
Base.expm1(x::Float32x) = convert(Float32x, expm1(Float64(x)))

# Logarithmic functions
Base.log(x::Float32x) = begin
    if isnan(x) || (isinf(x) && x.significand > 0)
        return x
    end
    if iszero(x)
        return Float32x(-Inf32, Int32(0))
    end
    if signbit(x)
        return nan(Float32x)
    end
    # log(s * 2^e) = log(s) + e * log(2)
    Float32x(log(x.significand) + x.exponent * log(2f0), Int32(0))
end

Base.log2(x::Float32x) = begin
    if isnan(x) || (isinf(x) && x.significand > 0)
        return x
    end
    if iszero(x)
        return Float32x(-Inf32, Int32(0))
    end
    if signbit(x)
        return nan(Float32x)
    end
    Float32x(log2(x.significand) + Float32(x.exponent), Int32(0))
end

Base.log10(x::Float32x) = log(x) / log(Float32x(10))
Base.log1p(x::Float32x) = log(one(Float32x) + x)

# Root functions
Base.sqrt(x::Float32x) = begin
    if isnan(x) || (isinf(x) && x.significand > 0) || iszero(x)
        return x
    end
    if signbit(x)
        return nan(Float32x)
    end
    
    # sqrt(s * 2^e) = sqrt(s) * 2^(e/2)
    if iseven(x.exponent)
        Float32x(sqrt(x.significand), x.exponent ÷ 2)
    else
        Float32x(sqrt(x.significand * 2f0), (x.exponent - 1) ÷ 2)
    end
end

Base.cbrt(x::Float32x) = convert(Float32x, cbrt(Float64(x)))

# Trigonometric functions
Base.sin(x::Float32x) = convert(Float32x, sin(Float64(x)))
Base.cos(x::Float32x) = convert(Float32x, cos(Float64(x)))
Base.tan(x::Float32x) = convert(Float32x, tan(Float64(x)))
Base.asin(x::Float32x) = convert(Float32x, asin(Float64(x)))
Base.acos(x::Float32x) = convert(Float32x, acos(Float64(x)))
Base.atan(x::Float32x) = convert(Float32x, atan(Float64(x)))

# Hyperbolic functions
Base.sinh(x::Float32x) = convert(Float32x, sinh(Float64(x)))
Base.cosh(x::Float32x) = convert(Float32x, cosh(Float64(x)))
Base.tanh(x::Float32x) = convert(Float32x, tanh(Float64(x)))
Base.asinh(x::Float32x) = convert(Float32x, asinh(Float64(x)))
Base.acosh(x::Float32x) = convert(Float32x, acosh(Float64(x)))
Base.atanh(x::Float32x) = convert(Float32x, atanh(Float64(x)))

# Utility functions
Base.ldexp(x::Float32x, n::Integer) = Float32x(x.significand, x.exponent + Int32(n))
Base.frexp(x::Float32x) = (Float64(x.significand) / 2, Int(x.exponent) + 1)
Base.modf(x::Float32x) = begin
    f64 = Float64(x)
    ipart, fpart = modf(f64)
    (convert(Float32x, fpart), convert(Float32x, ipart))
end

# ==================== Display and String Operations ====================

Base.show(io::IO, x::Float32x) = begin
    if isnan(x)
        print(io, "NaN")
    elseif isinf(x)
        print(io, signbit(x) ? "-Inf" : "Inf")
    elseif iszero(x)
        print(io, signbit(x) ? "-0.0" : "0.0")
    else
        # Show as significand × 2^exponent for clarity
        print(io, x.significand, " × 2^", x.exponent)
    end
end

Base.string(x::Float32x) = sprint(show, x)
Base.print(io::IO, x::Float32x) = show(io, x)

# ==================== Hashing ====================

Base.hash(x::Float32x, h::UInt) = hash(x.exponent, hash(x.significand, h))

end # module

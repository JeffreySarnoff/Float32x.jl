"""
    Float32xs

A module implementing an extended precision floating-point type using a Float32 significand
and Int32 exponent, providing operations similar to Float64.
"""
module Float32xs

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
    @inline function Float32x(s::Float32, e::Int32)
        # Handle special cases
        isnan(s) && return new(NaN32, Int32(0))
        isinf(s) && return new(s, Int32(0))
        iszero(s) && return new(0.0f0, Int32(0))
        
        # Normalize: ensure 1.0 <= |significand| < 2.0 for non-zero values
        # Using optimized bit manipulation for frexp
        sig_bits = reinterpret(UInt32, s)
        exp_bits = (sig_bits >> 23) & 0x000000ff
        
        if exp_bits == 0  # Subnormal number
            # Handle subnormal case
            sig, exp_adj = frexp(s)
            return new(Float32(sig * 2), e + Int32(exp_adj) - 1)
        else
            # Normal number - extract exponent directly
            exp_adj = Int32(exp_bits) - 127
            # Set exponent to 127 (biased 0) to get significand in [1,2)
            new_bits = (sig_bits & 0x807fffff) | 0x3f800000
            return new(reinterpret(Float32, new_bits), e + exp_adj)
        end
    end
end

# Convenience constructor
@inline Float32x(x::Real) = convert(Float32x, x)

# ==================== Constants ====================

const ZERO_FLOAT32X = Float32x(0.0f0, Int32(0))
const ONE_FLOAT32X = Float32x(1.0f0, Int32(0))
const INF_FLOAT32X = Float32x(Inf32, Int32(0))
const NEGINF_FLOAT32X = Float32x(-Inf32, Int32(0))
const NAN_FLOAT32X = Float32x(NaN32, Int32(0))
const EPS_FLOAT32X = Float32x(eps(Float32), Int32(0))

@inline Base.zero(::Type{Float32x}) = ZERO_FLOAT32X
@inline Base.one(::Type{Float32x}) = ONE_FLOAT32X
@inline inf(::Type{Float32x}) = INF_FLOAT32X
@inline nan(::Type{Float32x}) = NAN_FLOAT32X
@inline Base.eps(::Type{Float32x}) = EPS_FLOAT32X
@inline Base.typemin(::Type{Float32x}) = NEGINF_FLOAT32X
@inline Base.typemax(::Type{Float32x}) = INF_FLOAT32X

# ==================== Predicates ====================

@inline Base.iszero(x::Float32x) = iszero(x.significand)
@inline Base.isone(x::Float32x) = x.significand == 1.0f0 && x.exponent == 0
@inline Base.isinf(x::Float32x) = isinf(x.significand)
@inline Base.isnan(x::Float32x) = isnan(x.significand)
@inline Base.isfinite(x::Float32x) = isfinite(x.significand)
@inline Base.issubnormal(x::Float32x) = issubnormal(x.significand)
@inline Base.signbit(x::Float32x) = signbit(x.significand)

# ==================== Comparison Operations ====================

@inline Base.:(==)(x::Float32x, y::Float32x) = begin
    # Fast path for common cases
    x.significand == y.significand && x.exponent == y.exponent
end

@inline Base.:(!=)(x::Float32x, y::Float32x) = !(x == y)

@inline Base.isless(x::Float32x, y::Float32x) = begin
    # NaN handling
    (isnan(x) | isnan(y)) && return false
    
    # Infinity handling
    xinf = isinf(x.significand)
    yinf = isinf(y.significand)
    (xinf | yinf) && return x.significand < y.significand
    
    # Sign comparison
    xsign = signbit(x)
    ysign = signbit(y)
    xsign != ysign && return xsign
    
    # Exponent comparison
    if x.exponent != y.exponent
        return xsign ? x.exponent > y.exponent : x.exponent < y.exponent
    end
    
    # Significand comparison
    return x.significand < y.significand
end

@inline Base.:(<=)(x::Float32x, y::Float32x) = !(y < x)
@inline Base.:(<)(x::Float32x, y::Float32x) = isless(x, y)
@inline Base.:(>=)(x::Float32x, y::Float32x) = !(x < y)
@inline Base.:(>)(x::Float32x, y::Float32x) = y < x

@inline Base.isequal(x::Float32x, y::Float32x) = 
    isequal(x.significand, y.significand) & isequal(x.exponent, y.exponent)

# ==================== Conversions ====================

@inline Base.convert(::Type{Float32x}, x::Float32x) = x

@inline Base.convert(::Type{Float32x}, x::Integer) = 
    iszero(x) ? ZERO_FLOAT32X : Float32x(Float32(x), Int32(0))

@inline Base.convert(::Type{Float32x}, x::Float32) = 
    Float32x(x, Int32(0))

@inline Base.convert(::Type{Float32x}, x::Float64) = begin
    isnan(x) && return NAN_FLOAT32X
    isinf(x) && return x > 0 ? INF_FLOAT32X : NEGINF_FLOAT32X
    iszero(x) && return ZERO_FLOAT32X
    
    # Use optimized frexp for Float64
    sig, exp = frexp(x)
    Float32x(Float32(sig * 2), Int32(exp - 1))
end

@inline Base.convert(::Type{Float64}, x::Float32x) = 
    ldexp(Float64(x.significand), Int(x.exponent))

@inline Base.convert(::Type{Float32}, x::Float32x) = begin
    # Clamp exponent to Float32 range
    exp_clamped = clamp(x.exponent, Int32(-126), Int32(127))
    ldexp(x.significand, Int(exp_clamped))
end

@inline Base.Float64(x::Float32x) = convert(Float64, x)
@inline Base.Float32(x::Float32x) = convert(Float32, x)
@inline Base.float(x::Float32x) = x

Base.promote_rule(::Type{Float32x}, ::Type{<:Integer}) = Float32x
Base.promote_rule(::Type{Float32x}, ::Type{Float32}) = Float32x
Base.promote_rule(::Type{Float32x}, ::Type{Float64}) = Float32x

# ==================== Basic Arithmetic ====================

@inline Base.:(+)(x::Float32x, y::Float32x) = begin
    # Special value handling with short-circuit evaluation
    isnan(x.significand) && return x
    isnan(y.significand) && return y
    
    # Infinity handling
    xinf = isinf(x.significand)
    yinf = isinf(y.significand)
    if xinf | yinf
        (xinf & yinf & (signbit(x) != signbit(y))) && return NAN_FLOAT32X
        return xinf ? x : y
    end
    
    # Zero handling
    iszero(x.significand) && return y
    iszero(y.significand) && return x
    
    # Align exponents efficiently
    exp_diff = x.exponent - y.exponent
    
    if exp_diff > 24  # Beyond Float32 precision
        return x
    elseif exp_diff < -24
        return y
    elseif exp_diff > 0
        # y needs shifting
        aligned_y = ldexp(y.significand, -exp_diff)
        result = x.significand + aligned_y
        return Float32x(result, x.exponent)
    elseif exp_diff < 0
        # x needs shifting
        aligned_x = ldexp(x.significand, exp_diff)
        result = aligned_x + y.significand
        return Float32x(result, y.exponent)
    else
        # Same exponent
        result = x.significand + y.significand
        return Float32x(result, x.exponent)
    end
end

@inline Base.:(-)(x::Float32x) = Float32x(-x.significand, x.exponent)
@inline Base.:(-)(x::Float32x, y::Float32x) = x + (-y)

@inline Base.:(*)(x::Float32x, y::Float32x) = begin
    # Special value handling
    isnan(x.significand) && return x
    isnan(y.significand) && return y
    
    # Check for infinity or zero
    xinf = isinf(x.significand)
    yinf = isinf(y.significand)
    xzero = iszero(x.significand)
    yzero = iszero(y.significand)
    
    # Inf * 0 = NaN
    ((xinf & yzero) | (yinf & xzero)) && return NAN_FLOAT32X
    
    # Handle infinities
    (xinf | yinf) && return Float32x(x.significand * y.significand, Int32(0))
    
    # Handle zeros
    (xzero | yzero) && return ZERO_FLOAT32X
    
    # Normal multiplication
    Float32x(x.significand * y.significand, x.exponent + y.exponent)
end

@inline Base.:(/)(x::Float32x, y::Float32x) = begin
    # Special value handling
    isnan(x.significand) && return x
    isnan(y.significand) && return y
    
    yzero = iszero(y.significand)
    if yzero
        iszero(x.significand) && return NAN_FLOAT32X
        s = copysign(1.0f0, x.significand) * copysign(1.0f0, y.significand)
        return Float32x(s * Inf32, Int32(0))
    end
    
    yinf = isinf(y.significand)
    if yinf
        isinf(x.significand) && return NAN_FLOAT32X
        return ZERO_FLOAT32X
    end
    
    if isinf(x.significand)
        s = copysign(1.0f0, x.significand) * copysign(1.0f0, y.significand)
        return Float32x(s * Inf32, Int32(0))
    end
    
    iszero(x.significand) && return ZERO_FLOAT32X
    
    # Normal division
    Float32x(x.significand / y.significand, x.exponent - y.exponent)
end

@inline Base.:(^)(x::Float32x, n::Integer) = begin
    n == 0 && return ONE_FLOAT32X
    n == 1 && return x
    n == 2 && return x * x
    
    if n < 0
        return ONE_FLOAT32X / (x^(-n))
    end
    
    # Square-and-multiply algorithm
    result = ONE_FLOAT32X
    base = x
    m = n
    while m > 0
        isodd(m) && (result *= base)
        m >>= 1
        m > 0 && (base *= base)
    end
    result
end

Base.:(^)(x::Float32x, y::Float32x) = exp(y * log(x))

# ==================== Pre-arithmetic Operations ====================

@inline Base.abs(x::Float32x) = Float32x(abs(x.significand), x.exponent)
@inline Base.sign(x::Float32x) = Float32x(sign(x.significand), Int32(0))

@inline Base.copysign(x::Float32x, y::Float32x) = 
    Float32x(copysign(x.significand, y.significand), x.exponent)

@inline Base.flipsign(x::Float32x, y::Float32x) = 
    Float32x(flipsign(x.significand, y.significand), x.exponent)

@inline Base.nextfloat(x::Float32x) = begin
    isnan(x.significand) && return x
    (isinf(x.significand) & (x.significand > 0)) && return x
    Float32x(nextfloat(x.significand), x.exponent)
end

@inline Base.prevfloat(x::Float32x) = begin
    isnan(x.significand) && return x
    (isinf(x.significand) & (x.significand < 0)) && return x
    Float32x(prevfloat(x.significand), x.exponent)
end

# ==================== Rounding Operations ====================

Base.round(x::Float32x) = convert(Float32x, round(Float64(x)))
Base.floor(x::Float32x) = convert(Float32x, floor(Float64(x)))
Base.ceil(x::Float32x) = convert(Float32x, ceil(Float64(x)))
Base.trunc(x::Float32x) = convert(Float32x, trunc(Float64(x)))

@inline Base.significand(x::Float32x) = Float64(x.significand)
@inline Base.exponent(x::Float32x) = x.exponent

# ==================== Mathematical Functions ====================

# ---- Exponential functions ----
Base.exp(x::Float32x) = convert(Float32x, exp(Float64(x)))
Base.exp2(x::Float32x) = convert(Float32x, exp2(Float64(x)))
Base.exp10(x::Float32x) = convert(Float32x, exp10(Float64(x)))
Base.expm1(x::Float32x) = convert(Float32x, expm1(Float64(x)))

# ---- Logarithmic functions with optimizations ----
const LOG2_CONST = Float32(0.6931471805599453)

@inline Base.log(x::Float32x) = begin
    isnan(x.significand) && return x
    (isinf(x.significand) & (x.significand > 0)) && return x
    iszero(x.significand) && return NEGINF_FLOAT32X
    signbit(x) && return NAN_FLOAT32X
    
    # log(s * 2^e) = log(s) + e * log(2)
    # Compute in Float32 for consistency
    log_sig = log(x.significand)
    exp_contrib = x.exponent * LOG2_CONST
    Float32x(log_sig + exp_contrib, Int32(0))
end

@inline Base.log2(x::Float32x) = begin
    isnan(x.significand) && return x
    (isinf(x.significand) & (x.significand > 0)) && return x
    iszero(x.significand) && return NEGINF_FLOAT32X
    signbit(x) && return NAN_FLOAT32X
    
    # log2(s * 2^e) = log2(s) + e
    Float32x(log2(x.significand) + Float32(x.exponent), Int32(0))
end

@inline Base.log10(x::Float32x) = log(x) * Float32x(0.4342944819032518f0, Int32(0))
@inline Base.log1p(x::Float32x) = log(ONE_FLOAT32X + x)

# ---- Root functions with optimizations ----
@inline Base.sqrt(x::Float32x) = begin
    isnan(x.significand) && return x
    (isinf(x.significand) & (x.significand > 0)) && return x
    iszero(x.significand) && return x
    signbit(x) && return NAN_FLOAT32X
    
    # sqrt(s * 2^e) = sqrt(s) * 2^(e/2)
    # Handle odd exponents by adjusting significand
    if iseven(x.exponent)
        Float32x(sqrt(x.significand), x.exponent >> 1)
    else
        # Multiply significand by 2 and adjust exponent
        Float32x(sqrt(x.significand * 2.0f0), (x.exponent - Int32(1)) >> 1)
    end
end

Base.cbrt(x::Float32x) = convert(Float32x, cbrt(Float64(x)))

# ---- Trigonometric functions ----
Base.sin(x::Float32x) = convert(Float32x, sin(Float64(x)))
Base.cos(x::Float32x) = convert(Float32x, cos(Float64(x)))
Base.tan(x::Float32x) = convert(Float32x, tan(Float64(x)))
Base.asin(x::Float32x) = convert(Float32x, asin(Float64(x)))
Base.acos(x::Float32x) = convert(Float32x, acos(Float64(x)))
Base.atan(x::Float32x) = convert(Float32x, atan(Float64(x)))

# ---- Hyperbolic functions ----
Base.sinh(x::Float32x) = convert(Float32x, sinh(Float64(x)))
Base.cosh(x::Float32x) = convert(Float32x, cosh(Float64(x)))
Base.tanh(x::Float32x) = convert(Float32x, tanh(Float64(x)))
Base.asinh(x::Float32x) = convert(Float32x, asinh(Float64(x)))
Base.acosh(x::Float32x) = convert(Float32x, acosh(Float64(x)))
Base.atanh(x::Float32x) = convert(Float32x, atanh(Float64(x)))

# ---- Utility functions ----
@inline Base.ldexp(x::Float32x, n::Integer) = 
    Float32x(x.significand, x.exponent + Int32(n))

@inline Base.frexp(x::Float32x) = 
    (Float64(x.significand) * 0.5, Int(x.exponent) + 1)

Base.modf(x::Float32x) = begin
    f64 = Float64(x)
    ipart, fpart = modf(f64)
    (convert(Float32x, fpart), convert(Float32x, ipart))
end

# ==================== Display and String Operations ====================

Base.show(io::IO, x::Float32x) = begin
    if isnan(x.significand)
        print(io, "NaN")
    elseif isinf(x.significand)
        print(io, signbit(x) ? "-Inf" : "Inf")
    elseif iszero(x.significand)
        print(io, signbit(x) ? "-0.0" : "0.0")
    else
        # Show as decimal for better readability
        print(io, Float64(x))
    end
end

Base.string(x::Float32x) = sprint(show, x)
Base.print(io::IO, x::Float32x) = show(io, x)

# ==================== Hashing ====================

@inline Base.hash(x::Float32x, h::UInt) = 
    hash(x.exponent, hash(x.significand, h))

end # module


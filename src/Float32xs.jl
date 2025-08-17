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
import Base: eps, typemin, typemax, floatmin, floatmax
import Base: float, Float64, Float32, Float16, BigFloat
import Base: hash

# ==================== Type Definition ====================

"""
    Float32x <: AbstractFloat

Extended precision floating-point type using Float32 significand and Int32 exponent.
Represents: significand × 2^exponent where 1.0 <= |significand| < 2.0 for normalized values
"""
struct Float32x <: AbstractFloat
    significand::Float32
    exponent::Int32
    
    # Inner constructor with normalization
    @inline function Float32x(s::Float32, e::Int32)
        # Handle special cases
        isnan(s) && return new(NaN32, Int32(0))
        isinf(s) && return new(s, Int32(0))
        iszero(s) && return new(copysign(0.0f0, s), Int32(0))  # Preserve signed zero
        
        # Normalize: ensure 1.0 <= |significand| < 2.0 for non-zero values
        sig_bits = reinterpret(UInt32, s)
        exp_bits = (sig_bits >> 23) & 0x000000ff
        
        if exp_bits == 0  # Subnormal number
            # Need full frexp for subnormals
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

# Exponent limits for Float32 and Float64
const FLOAT32_MAX_EXP = 127
const FLOAT32_MIN_EXP = -126
const FLOAT64_MAX_EXP = 1023
const FLOAT64_MIN_EXP = -1022

# Float32x can represent much larger/smaller values than Float32
# The smallest normalized positive Float32x is 1.0 * 2^(typemin(Int32))
# The largest finite Float32x is just under 2.0 * 2^(typemax(Int32))
const FLOATMIN_FLOAT32X = Float32x(1.0f0, typemin(Int32))
const FLOATMAX_FLOAT32X = Float32x(prevfloat(2.0f0), typemax(Int32))

@inline Base.zero(::Type{Float32x}) = ZERO_FLOAT32X
@inline Base.one(::Type{Float32x}) = ONE_FLOAT32X
@inline inf(::Type{Float32x}) = INF_FLOAT32X
@inline nan(::Type{Float32x}) = NAN_FLOAT32X
@inline Base.eps(::Type{Float32x}) = EPS_FLOAT32X
@inline Base.typemin(::Type{Float32x}) = NEGINF_FLOAT32X
@inline Base.typemax(::Type{Float32x}) = INF_FLOAT32X
@inline Base.floatmin(::Type{Float32x}) = FLOATMIN_FLOAT32X
@inline Base.floatmax(::Type{Float32x}) = FLOATMAX_FLOAT32X

# ==================== Predicates ====================

@inline Base.iszero(x::Float32x) = iszero(x.significand)
@inline Base.isone(x::Float32x) = x.significand == 1.0f0 && x.exponent == 0
@inline Base.isinf(x::Float32x) = isinf(x.significand)
@inline Base.isnan(x::Float32x) = isnan(x.significand)
@inline Base.isfinite(x::Float32x) = isfinite(x.significand)
@inline Base.issubnormal(x::Float32x) = false  # Float32x values are always normalized
@inline Base.signbit(x::Float32x) = signbit(x.significand)

# ==================== Comparison Operations ====================

@inline Base.:(==)(x::Float32x, y::Float32x) = begin
    # NaN != NaN
    (isnan(x.significand) | isnan(y.significand)) && return false
    # -0.0 == 0.0
    (iszero(x.significand) & iszero(y.significand)) && return true
    # Compare both fields
    x.significand == y.significand && x.exponent == y.exponent
end

@inline Base.:(!=)(x::Float32x, y::Float32x) = !(x == y)

@inline Base.isless(x::Float32x, y::Float32x) = begin
    # NaN handling
    (isnan(x.significand) | isnan(y.significand)) && return false
    
    # Infinity handling
    xinf = isinf(x.significand)
    yinf = isinf(y.significand)
    (xinf | yinf) && return x.significand < y.significand
    
    # Zero handling: -0.0 == 0.0
    xzero = iszero(x.significand)
    yzero = iszero(y.significand)
    
    # Both zero - they're equal
    (xzero & yzero) && return false
    
    # One is zero - handle specially
    if xzero
        # x is zero, y is not
        # 0 < positive, 0 > negative
        return !signbit(y)
    elseif yzero
        # y is zero, x is not
        # negative < 0, positive > 0
        return signbit(x)
    end
    
    # Neither is zero - continue with normal comparison
    # Sign comparison
    xsign = signbit(x)
    ysign = signbit(y)
    xsign != ysign && return xsign
    
    # Same sign: compare exponents
    if x.exponent != y.exponent
        return xsign ? x.exponent > y.exponent : x.exponent < y.exponent
    end
    
    # Same sign and exponent: compare significands
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
    iszero(x) && return copysign(ZERO_FLOAT32X, x)
    
    # Use optimized frexp for Float64
    sig, exp = frexp(x)
    Float32x(Float32(sig * 2), Int32(exp - 1))
end

@inline Base.convert(::Type{Float32x}, x::BigFloat) = begin
    isnan(x) && return NAN_FLOAT32X
    isinf(x) && return x > 0 ? INF_FLOAT32X : NEGINF_FLOAT32X
    iszero(x) && return signbit(x) ? Float32x(-0.0f0, Int32(0)) : ZERO_FLOAT32X
    
    # Extract significand and exponent from BigFloat
    # BigFloat maintains precision, so we need to carefully extract
    sig_big, exp_big = frexp(x)
    
    # Convert significand to Float32 (may lose precision)
    sig_f32 = Float32(sig_big * 2)  # Scale to [1, 2) range
    
    # Check for exponent overflow/underflow
    if exp_big - 1 > typemax(Int32)
        return signbit(x) ? NEGINF_FLOAT32X : INF_FLOAT32X
    elseif exp_big - 1 < typemin(Int32)
        return signbit(x) ? Float32x(-0.0f0, Int32(0)) : ZERO_FLOAT32X
    end
    
    Float32x(sig_f32, Int32(exp_big - 1))
end

@inline Base.convert(::Type{Float64}, x::Float32x) = begin
    # Handle special cases
    isnan(x.significand) && return NaN64
    isinf(x.significand) && return Float64(x.significand)
    iszero(x.significand) && return copysign(0.0, x.significand)
    
    # Check for overflow/underflow
    total_exp = Int(x.exponent)
    
    # Float64 can handle much larger range than Float32x
    if total_exp > FLOAT64_MAX_EXP
        return copysign(Inf64, x.significand)
    elseif total_exp < FLOAT64_MIN_EXP - 52  # Account for significand bits
        return copysign(0.0, x.significand)
    end
    
    # Safe conversion
    ldexp(Float64(x.significand), total_exp)
end

@inline Base.convert(::Type{Float32}, x::Float32x) = begin
    # Handle special cases
    isnan(x.significand) && return NaN32
    isinf(x.significand) && return x.significand
    iszero(x.significand) && return copysign(0.0f0, x.significand)
    
    # Check for overflow/underflow
    total_exp = Int(x.exponent)
    
    if total_exp > FLOAT32_MAX_EXP
        return copysign(Inf32, x.significand)
    elseif total_exp < FLOAT32_MIN_EXP - 23  # Account for significand bits
        return copysign(0.0f0, x.significand)
    end
    
    # Safe conversion
    ldexp(x.significand, total_exp)
end

@inline Base.convert(::Type{BigFloat}, x::Float32x) = begin
    # Handle special cases
    isnan(x.significand) && return BigFloat(NaN)
    isinf(x.significand) && return BigFloat(x.significand)
    iszero(x.significand) && return BigFloat(x.significand)  # Preserves sign
    
    # Convert to BigFloat with full precision
    # x = significand * 2^exponent
    sig_big = BigFloat(x.significand)
    
    # Use ldexp for exact scaling
    ldexp(sig_big, Int(x.exponent))
end

@inline Base.Float64(x::Float32x) = convert(Float64, x)
@inline Base.Float32(x::Float32x) = convert(Float32, x)
@inline Base.BigFloat(x::Float32x) = convert(BigFloat, x)
@inline Base.float(x::Float32x) = x

Base.promote_rule(::Type{Float32x}, ::Type{<:Integer}) = Float32x
Base.promote_rule(::Type{Float32x}, ::Type{Float32}) = Float32x
Base.promote_rule(::Type{Float32x}, ::Type{Float64}) = Float32x
Base.promote_rule(::Type{Float32x}, ::Type{BigFloat}) = BigFloat

# ==================== Basic Arithmetic ====================

@inline Base.:(+)(x::Float32x, y::Float32x) = begin
    # Special value handling
    isnan(x.significand) && return x
    isnan(y.significand) && return y
    
    # Infinity handling
    xinf = isinf(x.significand)
    yinf = isinf(y.significand)
    if xinf | yinf
        (xinf & yinf & (signbit(x) != signbit(y))) && return NAN_FLOAT32X
        return xinf ? x : y
    end
    
    # Zero handling - preserve sign for -0.0 + 0.0 = 0.0
    xzero = iszero(x.significand)
    yzero = iszero(y.significand)
    xzero && return y
    yzero && return x
    
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
    
    # Check for infinity and zero
    xinf = isinf(x.significand)
    yinf = isinf(y.significand)
    xzero = iszero(x.significand)
    yzero = iszero(y.significand)
    
    # Inf * 0 = NaN
    ((xinf & yzero) | (yinf & xzero)) && return NAN_FLOAT32X
    
    # Handle infinities
    (xinf | yinf) && return Float32x(x.significand * y.significand, Int32(0))
    
    # Handle zeros - preserve sign
    (xzero | yzero) && return Float32x(x.significand * y.significand, Int32(0))
    
    # Check for potential overflow in exponent addition
    exp_sum = Int64(x.exponent) + Int64(y.exponent)
    if exp_sum > typemax(Int32)
        return Float32x(copysign(Inf32, x.significand * y.significand), Int32(0))
    elseif exp_sum < typemin(Int32)
        return Float32x(copysign(0.0f0, x.significand * y.significand), Int32(0))
    end
    
    # Normal multiplication
    Float32x(x.significand * y.significand, Int32(exp_sum))
end

@inline Base.:(/)(x::Float32x, y::Float32x) = begin
    # Special value handling
    isnan(x.significand) && return x
    isnan(y.significand) && return y
    
    yzero = iszero(y.significand)
    if yzero
        iszero(x.significand) && return NAN_FLOAT32X  # 0/0 = NaN
        s = copysign(1.0f0, x.significand) * copysign(1.0f0, y.significand)
        return Float32x(s * Inf32, Int32(0))
    end
    
    yinf = isinf(y.significand)
    if yinf
        isinf(x.significand) && return NAN_FLOAT32X  # Inf/Inf = NaN
        # finite/Inf = 0 with appropriate sign
        s = copysign(1.0f0, x.significand) * copysign(1.0f0, y.significand)
        return Float32x(copysign(0.0f0, s), Int32(0))
    end
    
    if isinf(x.significand)
        s = copysign(1.0f0, x.significand) * copysign(1.0f0, y.significand)
        return Float32x(s * Inf32, Int32(0))
    end
    
    if iszero(x.significand)
        # 0/finite = 0 with appropriate sign
        s = copysign(1.0f0, x.significand) * copysign(1.0f0, y.significand)
        return Float32x(copysign(0.0f0, s), Int32(0))
    end
    
    # Check for potential overflow/underflow in exponent subtraction
    exp_diff = Int64(x.exponent) - Int64(y.exponent)
    if exp_diff > typemax(Int32)
        return Float32x(copysign(Inf32, x.significand / y.significand), Int32(0))
    elseif exp_diff < typemin(Int32)
        return Float32x(copysign(0.0f0, x.significand / y.significand), Int32(0))
    end
    
    # Normal division
    Float32x(x.significand / y.significand, Int32(exp_diff))
end

@inline Base.:(^)(x::Float32x, n::Integer) = begin
    n == 0 && return ONE_FLOAT32X
    n == 1 && return x
    n == 2 && return x * x
    
    if n < 0
        return ONE_FLOAT32X / (x^(-n))
    end
    
    # Square-and-multiply algorithm with overflow checking
    result = ONE_FLOAT32X
    base = x
    m = n
    while m > 0
        if isodd(m)
            result = result * base
            # Check for overflow/special values
            (isinf(result.significand) | isnan(result.significand)) && return result
        end
        m >>= 1
        if m > 0
            base = base * base
            # Check for overflow/special values
            (isinf(base.significand) | isnan(base.significand)) && (m > 0 && isodd(m)) && return base
        end
    end
    result
end

Base.:(^)(x::Float32x, y::Float32x) = begin
    # Handle special cases properly
    isnan(x.significand) && return x
    isnan(y.significand) && return y
    
    # 1^y = 1 for any y (including Inf and NaN is already handled)
    isone(x) && return ONE_FLOAT32X
    
    # x^0 = 1 for any x != 0
    iszero(y) && return ONE_FLOAT32X
    
    # 0^y handling
    if iszero(x)
        if signbit(y)
            # 0^(-y) = Inf for y > 0
            return INF_FLOAT32X
        else
            # 0^y = 0 for y > 0
            return ZERO_FLOAT32X
        end
    end
    
    # x^1 = x
    isone(y) && return x
    
    # Negative base with non-integer exponent = NaN
    if signbit(x)
        y_float = Float64(y)
        if y_float != trunc(y_float)
            return NAN_FLOAT32X
        end
    end
    
    exp(y * log(abs(x))) * (signbit(x) && isodd(Int(trunc(Float64(y)))) ? -1 : 1)
end

# ==================== Pre-arithmetic Operations ====================

@inline Base.abs(x::Float32x) = Float32x(abs(x.significand), x.exponent)
@inline Base.sign(x::Float32x) = begin
    isnan(x.significand) && return x
    iszero(x.significand) && return x  # Preserve signed zero
    Float32x(copysign(1.0f0, x.significand), Int32(0))
end

@inline Base.copysign(x::Float32x, y::Float32x) = 
    Float32x(copysign(x.significand, y.significand), x.exponent)

@inline Base.flipsign(x::Float32x, y::Float32x) = 
    Float32x(flipsign(x.significand, y.significand), x.exponent)

@inline Base.nextfloat(x::Float32x) = begin
    isnan(x.significand) && return x
    (isinf(x.significand) & (x.significand > 0)) && return x
    
    next_sig = nextfloat(x.significand)
    # Check if we wrapped around (went from just under 2.0 to 2.0)
    if next_sig >= 2.0f0
        Float32x(1.0f0, x.exponent + Int32(1))
    else
        Float32x(next_sig, x.exponent)
    end
end

@inline Base.prevfloat(x::Float32x) = begin
    isnan(x.significand) && return x
    (isinf(x.significand) & (x.significand < 0)) && return x
    
    prev_sig = prevfloat(x.significand)
    # Check if we wrapped around (went from 1.0 to just under 1.0)
    if x.significand == 1.0f0 && prev_sig < 1.0f0
        Float32x(prevfloat(2.0f0), x.exponent - Int32(1))
    else
        Float32x(prev_sig, x.exponent)
    end
end

# ==================== Rounding Operations ====================

Base.round(x::Float32x) = begin
    isnan(x.significand) | isinf(x.significand) && return x
    f64 = Float64(x)
    # Check if the value is too large to round
    abs(f64) >= 2^53 && return x
    convert(Float32x, round(f64))
end

Base.floor(x::Float32x) = begin
    isnan(x.significand) | isinf(x.significand) && return x
    f64 = Float64(x)
    abs(f64) >= 2^53 && return x
    convert(Float32x, floor(f64))
end

Base.ceil(x::Float32x) = begin
    isnan(x.significand) | isinf(x.significand) && return x
    f64 = Float64(x)
    abs(f64) >= 2^53 && return x
    convert(Float32x, ceil(f64))
end

Base.trunc(x::Float32x) = begin
    isnan(x.significand) | isinf(x.significand) && return x
    f64 = Float64(x)
    abs(f64) >= 2^53 && return x
    convert(Float32x, trunc(f64))
end

# ---- More robust significand and exponent functions ----
@inline Base.significand(x::Float32x) = begin
    # Returns the significand as a Float64 in the range [1, 2) for normalized values
    # Special cases return the appropriate values
    isnan(x.significand) && return NaN64
    isinf(x.significand) && return Float64(x.significand)
    iszero(x.significand) && return copysign(0.0, x.significand)
    Float64(x.significand)
end

@inline Base.exponent(x::Float32x) = begin
    # Returns the binary exponent of x
    # For special values, follows Julia's convention
    isnan(x.significand) && throw(DomainError(x, "Cannot take exponent of NaN"))
    isinf(x.significand) && throw(DomainError(x, "Cannot take exponent of Inf"))
    iszero(x.significand) && throw(DomainError(x, "Cannot take exponent of zero"))
    x.exponent
end

# ---- Robust floatmin and floatmax functions ----
@inline Base.floatmin(x::Float32x) = floatmin(Float32x)
@inline Base.floatmax(x::Float32x) = floatmax(Float32x)

# ==================== Mathematical Functions ====================

# ---- Exponential functions ----
Base.exp(x::Float32x) = begin
    isnan(x.significand) && return x
    isinf(x.significand) && return x.significand > 0 ? x : ZERO_FLOAT32X
    # Overflow/underflow handling
    f64 = Float64(x)
    f64 > 88.72 && return INF_FLOAT32X  # exp overflow threshold
    f64 < -87.33 && return ZERO_FLOAT32X  # exp underflow threshold
    convert(Float32x, exp(f64))
end

Base.exp2(x::Float32x) = begin
    isnan(x.significand) && return x
    isinf(x.significand) && return x.significand > 0 ? x : ZERO_FLOAT32X
    f64 = Float64(x)
    f64 > 127 && return INF_FLOAT32X
    f64 < -126 && return ZERO_FLOAT32X
    convert(Float32x, exp2(f64))
end

Base.exp10(x::Float32x) = begin
    isnan(x.significand) && return x
    isinf(x.significand) && return x.significand > 0 ? x : ZERO_FLOAT32X
    f64 = Float64(x)
    f64 > 38.5 && return INF_FLOAT32X
    f64 < -37.9 && return ZERO_FLOAT32X
    convert(Float32x, exp10(f64))
end

Base.expm1(x::Float32x) = begin
    isnan(x.significand) && return x
    isinf(x.significand) && return x.significand > 0 ? x : NEGINF_FLOAT32X
    convert(Float32x, expm1(Float64(x)))
end

# ---- Logarithmic functions with optimizations ----
const LOG2_CONST = Float32(0.6931471805599453)

@inline Base.log(x::Float32x) = begin
    isnan(x.significand) && return x
    (isinf(x.significand) & (x.significand > 0)) && return x
    iszero(x.significand) && return NEGINF_FLOAT32X
    signbit(x) && return NAN_FLOAT32X
    
    # log(s * 2^e) = log(s) + e * log(2)
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

@inline Base.log10(x::Float32x) = begin
    isnan(x.significand) && return x
    (isinf(x.significand) & (x.significand > 0)) && return x
    iszero(x.significand) && return NEGINF_FLOAT32X
    signbit(x) && return NAN_FLOAT32X
    
    log(x) * Float32x(0.4342944819032518f0, Int32(0))
end

@inline Base.log1p(x::Float32x) = begin
    isnan(x.significand) && return x
    # log1p(-1) = -Inf
    x == Float32x(-1.0f0, Int32(0)) && return NEGINF_FLOAT32X
    # log1p(x) for x < -1 = NaN
    x < Float32x(-1.0f0, Int32(0)) && return NAN_FLOAT32X
    isinf(x.significand) && x.significand > 0 && return x
    
    log(ONE_FLOAT32X + x)
end

# ---- Root functions with optimizations ----
@inline Base.sqrt(x::Float32x) = begin
    isnan(x.significand) && return x
    (isinf(x.significand) & (x.significand > 0)) && return x
    iszero(x.significand) && return x  # Preserve signed zero
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

Base.cbrt(x::Float32x) = begin
    isnan(x.significand) | isinf(x.significand) | iszero(x.significand) && return x
    convert(Float32x, cbrt(Float64(x)))
end

# ---- Trigonometric functions ----
Base.sin(x::Float32x) = begin
    isnan(x.significand) && return x
    isinf(x.significand) && return NAN_FLOAT32X  # sin(±Inf) = NaN
    convert(Float32x, sin(Float64(x)))
end

Base.cos(x::Float32x) = begin
    isnan(x.significand) && return x
    isinf(x.significand) && return NAN_FLOAT32X  # cos(±Inf) = NaN
    convert(Float32x, cos(Float64(x)))
end

Base.tan(x::Float32x) = begin
    isnan(x.significand) && return x
    isinf(x.significand) && return NAN_FLOAT32X  # tan(±Inf) = NaN
    convert(Float32x, tan(Float64(x)))
end

Base.asin(x::Float32x) = begin
    isnan(x.significand) && return x
    abs(x) > ONE_FLOAT32X && return NAN_FLOAT32X  # asin(|x|>1) = NaN
    convert(Float32x, asin(Float64(x)))
end

Base.acos(x::Float32x) = begin
    isnan(x.significand) && return x
    abs(x) > ONE_FLOAT32X && return NAN_FLOAT32X  # acos(|x|>1) = NaN
    convert(Float32x, acos(Float64(x)))
end

Base.atan(x::Float32x) = begin
    isnan(x.significand) && return x
    isinf(x.significand) && return Float32x(copysign(Float32(π/2), x.significand), Int32(0))
    convert(Float32x, atan(Float64(x)))
end

# ---- Hyperbolic functions ----
Base.sinh(x::Float32x) = begin
    isnan(x.significand) | isinf(x.significand) && return x
    f64 = Float64(x)
    abs(f64) > 88.72 && return copysign(INF_FLOAT32X, x)  # sinh overflow
    convert(Float32x, sinh(f64))
end

Base.cosh(x::Float32x) = begin
    isnan(x.significand) && return x
    isinf(x.significand) && return INF_FLOAT32X  # cosh(±Inf) = Inf
    f64 = Float64(x)
    abs(f64) > 88.72 && return INF_FLOAT32X  # cosh overflow
    convert(Float32x, cosh(f64))
end

Base.tanh(x::Float32x) = begin
    isnan(x.significand) && return x
    isinf(x.significand) && return Float32x(copysign(1.0f0, x.significand), Int32(0))
    convert(Float32x, tanh(Float64(x)))
end

Base.asinh(x::Float32x) = begin
    isnan(x.significand) | isinf(x.significand) && return x
    convert(Float32x, asinh(Float64(x)))
end

Base.acosh(x::Float32x) = begin
    isnan(x.significand) && return x
    x < ONE_FLOAT32X && return NAN_FLOAT32X  # acosh(x<1) = NaN
    isinf(x.significand) && x.significand > 0 && return x
    convert(Float32x, acosh(Float64(x)))
end

Base.atanh(x::Float32x) = begin
    isnan(x.significand) && return x
    abs_x = abs(x)
    abs_x > ONE_FLOAT32X && return NAN_FLOAT32X  # atanh(|x|>1) = NaN
    abs_x == ONE_FLOAT32X && return copysign(INF_FLOAT32X, x)  # atanh(±1) = ±Inf
    convert(Float32x, atanh(Float64(x)))
end

# ---- Utility functions ----
@inline Base.ldexp(x::Float32x, n::Integer) = begin
    # ldexp(x, n) = x * 2^n
    # Handle special cases
    isnan(x.significand) | isinf(x.significand) | iszero(x.significand) && return x
    
    # Check for exponent overflow/underflow
    new_exp = Int64(x.exponent) + Int64(n)
    if new_exp > typemax(Int32)
        return copysign(INF_FLOAT32X, x)
    elseif new_exp < typemin(Int32) + 200  # Leave some room for normalization
        return copysign(ZERO_FLOAT32X, x)
    end
    
    Float32x(x.significand, Int32(new_exp))
end

# More robust ldexp that handles Float32 significand and separate integer exponent
@inline Base.ldexp(s::Real, e::Integer, ::Type{Float32x}) = begin
    # Create Float32x from separate significand and exponent
    # This is useful for reconstructing values from frexp
    sf = Float32(s)
    isnan(sf) && return NAN_FLOAT32X
    isinf(sf) && return sf > 0 ? INF_FLOAT32X : NEGINF_FLOAT32X
    iszero(sf) && return copysign(ZERO_FLOAT32X, sf)
    
    # Normalize s to [1, 2) range if needed
    sig, exp_adj = frexp(sf)
    total_exp = Int64(e) + Int64(exp_adj)
    
    # Check for overflow/underflow
    if total_exp > typemax(Int32)
        return copysign(INF_FLOAT32X, sf)
    elseif total_exp < typemin(Int32) + 200
        return copysign(ZERO_FLOAT32X, sf)
    end
    
    Float32x(Float32(sig * 2), Int32(total_exp - 1))
end

@inline Base.frexp(x::Float32x) = begin
    # frexp returns (f, e) such that x = f * 2^e with 0.5 <= |f| < 1
    # Special cases
    isnan(x.significand) && return (NaN64, 0)
    isinf(x.significand) && return (Float64(x.significand), 0)
    iszero(x.significand) && return (copysign(0.0, x.significand), 0)
    
    # For Float32x with significand in [1, 2), we need to return f in [0.5, 1)
    # x = significand * 2^exponent where significand ∈ [1, 2)
    # We want: x = f * 2^e where f ∈ [0.5, 1)
    # So: f = significand/2 and e = exponent + 1
    (Float64(x.significand) * 0.5, Int(x.exponent) + 1)
end

Base.modf(x::Float32x) = begin
    isnan(x.significand) | isinf(x.significand) && return (copysign(ZERO_FLOAT32X, x), x)
    f64 = Float64(x)
    # Check if integer part would overflow Float64
    abs(f64) >= 2^53 && return (copysign(ZERO_FLOAT32X, x), x)
    
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
        # But handle potential overflow in conversion
        if abs(x.exponent) < 300
            f64 = Float64(x)
            if isfinite(f64)
                print(io, f64)
            else
                # Fallback to showing components
                print(io, x.significand, " × 2^", x.exponent)
            end
        else
            # Very large exponent - show in component form
            print(io, x.significand, " × 2^", x.exponent)
        end
    end
end

Base.string(x::Float32x) = sprint(show, x)
Base.print(io::IO, x::Float32x) = show(io, x)

# ==================== Hashing ====================

@inline Base.hash(x::Float32x, h::UInt) = 
    hash(x.exponent, hash(x.significand, h))

end # module


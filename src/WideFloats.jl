module WideFloats

export WideFloat, is_widefloat
export Float16x, Float32x, Float64x
  export zero, one, inf, nan, eps
export iszero, isone, isinf, isnan, isfinite, issubnormal, signbit
export abs, sign, copysign, flipsign, nextfloat, prevfloat
export round, floor, ceil, trunc, significand, exponent
export sqrt, cbrt, exp, exp2, exp10, expm1
export log, log2, log10, log1p
export sin, cos, tan, asin, acos, atan
export sinh, cosh, tanh, asinh, acosh, atanh
export ldexp, frexp, modf
export faa, faa_compensated
export canonical

import Base: convert
import Base: UInt32, UInt64, UInt128

import Base: +, -, *, /, ^, %, ÷, fma
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
import Base: sign, significand, exponent
import Base: eps, typemin, typemax, floatmin, floatmax
import Base: float, Float64, Float32, Float16, BigFloat
import Base: hash

abstract type WideFloat <: AbstractFloat end

primitive type Float16x <: WideFloat  32 end
primitive type Float32x <: WideFloat  64 end
primitive type Float64x <: WideFloat 128 end

is_widefloat(x) = false
is_widefloat(T::Type{<:AbstractFloat}) = T <: WideFloat
is_widefloat(x::T) where {T} = T <: WideFloat

const FX   = Union{Float16x, Float32x, Float64x}
const UINT = Union{UInt32, UInt64, UInt128}

Base.convert(::Type{Float16x}, x::UInt32)  = reinterpret(Float16x, x)
Base.convert(::Type{Float32x}, x::UInt64)  = reinterpret(Float32x, x)
Base.convert(::Type{Float64x}, x::UInt128) = reinterpret(Float64x, x)

Float16x(x::UInt32)  = convert(Float16x, x)
Float32x(x::UInt64)  = convert(Float16x, x)
Float64x(x::UInt128) = convert(Float16x, x)

Base.convert(::Type{UInt32},  x::Float16x) = reinterpret(UInt32, x)
Base.convert(::Type{UInt64},  x::Float32x) = reinterpret(UInt64, x)
Base.convert(::Type{UInt128}, x::Float64x) = reinterpret(UInt128, x)

Base.UInt32(x::Float16x)  = convert(UInt32, x)
Base.Uint64(x::Float32x)  = convert(UInt64, x)
Base.UInt128(x::Float64x) = convert(UInt128, x)

# UnsignedHalf SignedHalf, Uint Float
for (U, Hu, Hs, F) in ((:UInt32,  :UInt16, :Int16, :Float16x), 
                       (:UInt64,  :UInt32, :Int32, :Float32x), 
                       (:UInt128, :UInt64, :Int64, :Float64x))
  @eval begin
    halfbits = 8*sizeof($Hu)
    lowones  = ~zero($U) >> halfbits

          
    # enfold an FNxUIntN into (significand, exponent)
    enfold(@nospecialize(x::Union{$U, $F})) = x
    enfold(fr::$F, xp::$Hs) =
        (((reinterpret($H, fr) % $U) << halfbits)) | reinterpret($Hu, xp)

    enfold(frxp::Tuple{$F, $U}) = enfold(frxp[1], frxp[2])

    # unfold a UIntN into (significand, exponent)
    function unfold(x::$U)
        xhi = (x >> halfbits) % $U
        xlo = (reinterpret($U, x) & lowones) % $Hu
        sig = reinterpret($F, xhi)
        xp  = reinterpret($I, xlo)
        sig, xp
    end

    function implicit_significand(x::$U)
        xhi = (x >> halfbits) % $U
        reinterpret($F, xhi) # sig
    end
    function explicit_exponent(x::$U)
        xlo = (reinterpret($U, x) & lowones) % $Hu
        reinterpret($I, xlo) # xp
    end

    @inline function is_canonical(x::$F)
        sig = significand(x)
        
        # This version assumes finite values for maximum speed
        # Use when you know values are finite
        abs_sig = abs(sig)
        
        # Returns true for zero or [0.5, 1.0)
        # Note: This relies on floating-point comparison behavior
        (iszero(x) || !isfinite(x)) || ((abs_sig >= (one($F) / 2)) && (abs_sig < one($F)))
    end

  end # @eval
end

# U,              S,             Hu,          Hs,         F,     Fx 
# UnsignedCarrier SignedCarrier, UnsignedHalf SignedHalf, Float, Floatx

for (U, S, Hu, Hs, F, Fx) in (
    (:UInt32,  :Int32, :UInt16, :Int16, :Float16, :Float16x), 
    (:UInt64,  :Int64, :UInt32, :Int32, :Float32, :Float32x), 
    (:UInt128, :Int128, :UInt64, :Int64, :Float64, :Float64x))
  @eval begin
 
    zerox   = zero($F)
    posinfs = $F(Inf)
    neginfx = $F(-Inf)
    nanx    = $F(NaN)
  
    sign_bitmask = Base.sign_mask($F)
    exp_bitmask  = Base.exponent_mask($F)
    sig_bitmask  = Base.significand_mask($F)
    exp_bitshift = trailing_zeros(exp_mask)
    exp_biasm1   = Base.exponent_bias($F) - 0x01

    @inline function canonicalize(x::$F)
        # Handle zero
        iszero(x.significand) && return x
        
        # Handle non-finite
        if !isfinite(x.significand)
            return x  # NaN or Inf unchanged
        end
        
        # Get bits of Float32 significand
        bits = reinterpret(UInt32, x.significand)
        
        # Extract components
        sign_bit = bits & sign_bitmask
        exp_bits = (bits >> exp_bitshift) & exp_bitmask
        significand_bits = bits & sig_bitmask
        
        # Handle subnormal Float32 (exp_bits == 0)
        if exp_bits == 0x00
            # Use general method for subnormals
            return canonicalize(x)
        end
        
        # For normal values, adjust to canonical form
        # We want exponent field to represent [0.5, 1.0)
        # In IEEE 754: 0.5 = 2^(-1) has biased exponent 126
        # So we set exp_bits to 126
        
        # Current unbiased exponent: exp_bits - 127
        # New significand will have exponent -1 (biased: 126)
        
        canonical_bits = sign_bit | ($Hu(exp_biasm1) << exp_bitshift) | significand_bits
        canonical_sig = reinterpret($F, canonical_bits)
        
        # Adjust FloatNx exponent
        # Original: significand * 2^(exp_bits-127) * 2^(x.exponent)
        # New: significand * 2^(-1) * 2^(new_exponent)
        # So: new_exponent = x.exponent + (exp_bits - 127) + 1
        
        exp_adjustment = $Hs(exp_bits) - $Hs(exp_biasm1)
        total_exp = $S(x.exponent) + $S(exp_adjustment)
        
        # Check bounds
        if total_exp > typemax($S)
            return iszero(sign_bit) ? posinfx : neginfx
        elseif total_exp < typemin($S)
            return iszero(sign_bit) ? zerox : $F(-zerox, zerox)
        end
        
        canonical_exp = $S(total_exp)

        return $F(canonical_sig, canonical_exp)
    end

  end # @eval
end



@inline function iszero(x::Fx)
        x.significand == zero(Float32)
    end

    @inline function isone(x::Fx)
        x.significand == one(Float32)
    end

    @inline function isinf(x::Fx)
        isinf(x.significand)
    end

    @inline function isnan(x::Fx)
        isnan(x.significand)
    end

    @inline function isfinite(x::Fx)
        isfinite(x.significand)
    end

    @inline function issubnormal(x::Fx)
        issubnormal(x.significand)
    end

    @inline function signbit(x::Fx)
        signbit(x.significand)
    end
end
Float32x(x::UInt64) = reinterpret(Float32x, x)
Float32x(x::UInt64) = reinterpret(Float32x, x)

Base.UInt64(x::Float32x) = x

function Float32x(sig::Float32, xp::Int32)
    reinterpret(Float32x, enfold(sig, xp))
end

Float32x(sigxp::Tuple{Float32, Int32}) = Float32x(sigxp[1], sigxp[2])

Float32x(x::UInt64) = reinterpret(Float32x, x)

Float32x(sig::Float64, xp::Int64) = Float32x(Float32(sig), xp % Int32)
Float32x(sig::Float32, xp::Int64) = Float32x(sig, xp % Int32)
Float32x(sig::Float64, xp::Int32) = Float32x(Float32(sig), xp)

# rewrite the function to accept a Float32x value and, for zero and for non-finite values return the same value, and for finite values return the equivalent Float32x value in canonical form.  A Float32x value is in canonical form when the significand is either zero or the absolute value of the significand s in the clopen range [0.5, 1.0)

# Canonicalize a Float32x value
# Canonical form: significand is zero or |significand| ∈ [0.5, 1.0)

# Bit manipulation version for maximum performance
@inline function canonicalize(x::$F)
    sig = x.significand
    # Handle zero
    iszero(sig) && return x
    
    # Handle non-finite
    if !isfinite(sig)
        return x  # NaN or Inf unchanged
    end
    
    # Get bits of Float32 significand
    bits = reinterpret($Hu, sig)
    
    # Extract components
    sign_bit = bits & 0x80000000
    exp_bits = (bits >> 23) & 0x000000ff
    significand_bits = bits & 0x007fffff
    
    # Handle subnormal Float32 (exp_bits == 0)
    if exp_bits == zero($I)
        # Use general method for subnormals
        return canonicalize(x)
    end
    
    # For normal values, adjust to canonical form
    # We want exponent field to represent [0.5, 1.0)
    # In IEEE 754: 0.5 = 2^(-1) has biased exponent 126
    # So we set exp_bits to 126
    
    # Current unbiased exponent: exp_bits - 127
    # New significand will have exponent -1 (biased: 126)
    
    canonical_bits = sign_bit | (UInt32(126) << 23) | significand_bits
    canonical_sig = reinterpret(Float32, canonical_bits)
    
    # Adjust Float32x exponent
    # Original: significand * 2^(exp_bits-127) * 2^(x.exponent)
    # New: significand * 2^(-1) * 2^(new_exponent)
    # So: new_exponent = x.exponent + (exp_bits - 127) + 1
    
    exp_adjustment = Int32(exp_bits) - Int32(126)
    total_exp = Int64(x.exponent) + Int64(exp_adjustment)
    
    # Check bounds
    if total_exp > typemax(Int32)
        return sign_bit == 0x00 ? INF_FLOAT32X : NEGINF_FLOAT32X
    elseif total_exp < typemin(Int32)
        return sign_bit == 0x00 ? ZERO_FLOAT32X : Float32x(-0.0f0, Int32(0))
    end
    
    return Float32x(canonical_sig, Int32(total_exp))
end

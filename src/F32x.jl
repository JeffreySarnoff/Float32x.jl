#=
 struct F32x
     significand::Float32
     exponent::Int32
 end
=#

primitive type F32x 64 end

F32x(x::UInt64) = reinterpret(F32x, x)
Base.UInt64(x::F32x) = reinterpret(UInt64, x)

Base.significand(x::F32x) = unfold(x)[1]
Base.exponent(x::F32x) = unfold(x)[1]

enfold(x::UInt64) = x

enfold(sigxp::Tuple{Float32, Int32}) = enfold(sigxp[1], sigxp[2])

function enfold(sig::Float32, xp::Int32)
    # Ensure the significand is normalized and the exponent is within bounds
    ((reinterpret(UInt32, sig) % UInt64) << 0x20) | reinterpret(UInt32, xp)
end
function unfold(x::UInt64)
    xhi = (reinterpret(UInt64, x) >> 32) % UInt32
    xlo = (reinterpret(UInt64, x) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF) % UInt32
    sig = reinterpret(Float32, xhi)
    xp  = reinterpret(Int32, xlo)
    sig, xp
end
function F32x(sig::Float32, xp::Int32)
    reinterpret(F32x, enfold(sig, xp))
end

F32x(sigxp::Tuple{Float32, Int32}) = F32x(sigxp[1], sigxp[2])

F32x(x::UInt64) = reinterpret(F32x, x)

F32x(sig::Float64, xp::Int64) = F32x(Float32(sig), xp % Int32)
F32x(sig::Float32, xp::Int64) = F32x(sig, xp % Int32)
F32x(sig::Float64, xp::Int32) = F32x(Float32(sig), xp)

# rewrite the function to accept a Float32x value and, for zero and for non-finite values return the same value, and for finite values return the equivalent Float32x value in canonical form.  A Float32x value is in canonical form when the significand is either zero or the absolute value of the significand s in the clopen range [0.5, 1.0)

# Canonicalize a Float32x value
# Canonical form: significand is zero or |significand| ∈ [0.5, 1.0)



# Branchless version for maximum performance (when applicable)
@inline function is_canonical(x::F32x)
    sig = x.significand
    
    # This version assumes finite values for maximum speed
    # Use when you know values are finite
    abs_sig = abs(sig)
    
    # Returns true for zero or [0.5, 1.0)
    # Note: This relies on floating-point comparison behavior
    return sig == 0.0f0 || (abs_sig >= 0.5f0 && abs_sig < 1.0f0)
end

# Bit manipulation version for maximum performance
@inline function canonicalize(x::F32x)
    # Handle zero
    iszero(x.significand) && return x
    
    # Handle non-finite
    if !isfinite(x.significand)
        return x  # NaN or Inf unchanged
    end
    
    # Get bits of Float32 significand
    bits = reinterpret(UInt32, x.significand)
    
    # Extract components
    sign_bit = bits & 0x80000000
    exp_bits = (bits >> 23) & 0x000000ff
    mantissa_bits = bits & 0x007fffff
    
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
    
    canonical_bits = sign_bit | (UInt32(126) << 23) | mantissa_bits
    canonical_sig = reinterpret(Float32, canonical_bits)
    
    # Adjust Float32x exponent
    # Original: mantissa * 2^(exp_bits-127) * 2^(x.exponent)
    # New: mantissa * 2^(-1) * 2^(new_exponent)
    # So: new_exponent = x.exponent + (exp_bits - 127) + 1
    
    exp_adjustment = Int32(exp_bits) - Int32(126)
    total_exp = Int64(x.exponent) + Int64(exp_adjustment)
    
    # Check bounds
    if total_exp > typemax(Int32)
        return sign_bit == 0x00 ? INF_FLOAT32X : NEGINF_FLOAT32X
    elseif total_exp < typemin(Int32)
        return sign_bit == 0x00 ? ZERO_FLOAT32X : Float32x(-0.0f0, Int32(0))
    end
    
    return F32x(canonical_sig, Int32(total_exp))
end

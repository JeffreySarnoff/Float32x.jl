# Check if a Float32x value is in canonical form
# Canonical: significand is zero or |significand| ∈ [0.5, 1.0)
@inline function is_canonical(x::Float32x)
    sig = x.significand
    
    # Zero is canonical (exponent doesn't matter)
    iszero(sig) && return true
    
    # NaN and Inf are considered canonical (pass through unchanged)
    !isfinite(sig) && return true
    
    # For finite non-zero values, check if |significand| ∈ [0.5, 1.0)
    abs_sig = abs(sig)
    return 0.5f0 <= abs_sig < 1.0f0
end

# Optimized version using bit manipulation
@inline function is_canonical_bits(x::Float32x)
    sig = x.significand
    
    # Fast zero check
    sig == 0.0f0 && return true
    
    # Get bits
    bits = reinterpret(UInt32, sig)
    
    # Extract exponent field (bits 23-30)
    exp_bits = (bits >> 23) & 0x000000ff
    
    # Special values check
    if exp_bits == 0xff
        return true  # NaN or Inf are canonical
    end
    
    # Subnormal check (exp_bits == 0)
    if exp_bits == 0x00
        # Subnormal Float32 values are less than 0.5, so not canonical
        # unless they're zero (already checked above)
        return false
    end
    
    # For normal values, check if exponent represents [0.5, 1.0)
    # 0.5 = 2^(-1) has biased exponent 126
    # Values < 1.0 have biased exponent < 127
    # So canonical range is exp_bits == 126
    return exp_bits == 0x7e  # 0x7e = 126
end

# Ultra-fast version with minimal branches
@inline function is_canonical_fast(x::Float32x)
    sig = x.significand
    
    # Handle special cases together
    (sig == 0.0f0 || !isfinite(sig)) && return true
    
    # For finite non-zero, just check range
    abs_sig = abs(sig)
    return 0.5f0 <= abs_sig && abs_sig < 1.0f0
end

# Branchless version for maximum performance (when applicable)
@inline function is_canonical_branchless(x::Float32x)
    sig = x.significand
    
    # This version assumes finite values for maximum speed
    # Use when you know values are finite
    abs_sig = abs(sig)
    
    # Returns true for zero or [0.5, 1.0)
    # Note: This relies on floating-point comparison behavior
    return sig == 0.0f0 || (abs_sig >= 0.5f0 && abs_sig < 1.0f0)
end

# Comprehensive version with detailed checks
@inline function is_canonical_comprehensive(x::Float32x)
    sig = x.significand
    
    # Step 1: Check zero (most common special case)
    if sig == 0.0f0 || sig == -0.0f0
        return true
    end
    
    # Step 2: Check non-finite
    if isnan(sig) || isinf(sig)
        return true
    end
    
    # Step 3: Check range for finite non-zero
    abs_sig = abs(sig)
    
    # Must be in [0.5, 1.0)
    # Using exact Float32 constants for precision
    return abs_sig >= 0.5f0 && abs_sig < 1.0f0
end

#=
       VALIDATIONS AND TESTS
=#


# Validation and testing function
function validate_is_canonical()
    println("Validating is_canonical functions:")
    println("=" ^ 50)
    
    # Test cases: (value, expected_result, description)
    test_cases = [
        (Float32x(0.0f0, Int32(0)), true, "Positive zero"),
        (Float32x(-0.0f0, Int32(0)), true, "Negative zero"),
        (Float32x(0.5f0, Int32(10)), true, "Lower bound inclusive"),
        (Float32x(0.99999f0, Int32(5)), true, "Upper bound exclusive"),
        (Float32x(1.0f0, Int32(5)), false, "Exactly 1.0"),
        (Float32x(1.5f0, Int32(5)), false, "Greater than 1.0"),
        (Float32x(0.4999f0, Int32(5)), false, "Just below 0.5"),
        (Float32x(-0.5f0, Int32(10)), true, "Negative canonical"),
        (Float32x(-0.99999f0, Int32(5)), true, "Negative canonical"),
        (Float32x(-1.0f0, Int32(5)), false, "Negative non-canonical"),
        (Float32x(NaN32, Int32(0)), true, "NaN"),
        (Float32x(Inf32, Int32(0)), true, "Positive infinity"),
        (Float32x(-Inf32, Int32(0)), true, "Negative infinity"),
    ]
    
    # Test all implementations
    implementations = [
        ("is_canonical", is_canonical),
        ("is_canonical_bits", is_canonical_bits),
        ("is_canonical_fast", is_canonical_fast),
        ("is_canonical_comprehensive", is_canonical_comprehensive),
    ]
    
    for (name, func) in implementations
        println("\nTesting $name:")
        all_passed = true
        
        for (value, expected, description) in test_cases
            result = func(value)
            passed = result == expected
            all_passed &= passed
            
            if !passed
                println("  FAIL: $description")
                println("    Value: sig=$(value.significand), exp=$(value.exponent)")
                println("    Expected: $expected, Got: $result")
            end
        end
        
        if all_passed
            println("  ✓ All tests passed")
        end
    end
    
    println("\n" * "=" ^ 50)
end

# Performance comparison
function benchmark_is_canonical()
    println("\nBenchmarking is_canonical implementations:")
    println("=" ^ 50)
    
    # Create test data
    n = 1000000
    test_data = Float32x[]
    
    # Mix of canonical and non-canonical values
    for i in 1:n
        if rand() < 0.5
            # Create canonical value
            sig = (0.5f0 + 0.49999f0 * rand(Float32)) * (rand() < 0.5 ? 1 : -1)
            push!(test_data, Float32x(sig, rand(Int32(-100):Int32(100))))
        else
            # Create non-canonical value
            sig = (1.0f0 + rand(Float32)) * (rand() < 0.5 ? 1 : -1)
            push!(test_data, Float32x(sig, rand(Int32(-100):Int32(100))))
        end
    end
    
    # Add special values
    push!(test_data, ZERO_FLOAT32X)
    push!(test_data, nan(Float32x))
    push!(test_data, inf(Float32x))
    
    # Time each implementation
    implementations = [
        ("is_canonical", is_canonical),
        ("is_canonical_bits", is_canonical_bits),
        ("is_canonical_fast", is_canonical_fast),
        ("is_canonical_branchless", is_canonical_branchless),
    ]
    
    for (name, func) in implementations
        # Warm up
        for x in test_data[1:min(100, end)]
            func(x)
        end
        
        # Time
        start_time = time_ns()
        count = 0
        for x in test_data
            count += func(x) ? 1 : 0
        end
        elapsed = (time_ns() - start_time) / 1e9
        
        println("$name:")
        println("  Time: $(round(elapsed, digits=4)) seconds")
        println("  Rate: $(round(length(test_data) / elapsed / 1e6, digits=2)) M checks/sec")
        println("  Canonical found: $count / $(length(test_data))")
    end
end

# Edge case testing
function test_edge_cases()
    println("\nTesting edge cases for is_canonical:")
    println("=" ^ 50)
    
    # Test subnormal Float32 values
    tiny = Float32(1e-40)  # Subnormal
    x_tiny = Float32x(tiny, Int32(0))
    println("Subnormal $(tiny): ", is_canonical(x_tiny))
    
    # Test exact boundary values
    boundary_values = [
        0.5f0,           # Exact lower bound
        nextfloat(0.5f0),  # Just above lower bound
        prevfloat(0.5f0),  # Just below lower bound
        prevfloat(1.0f0),  # Just below upper bound
        1.0f0,           # Exact upper bound
    ]
    
    for val in boundary_values
        x = Float32x(val, Int32(0))
        result = is_canonical(x)
        in_range = 0.5f0 <= val < 1.0f0
        println("Value $val: is_canonical=$result, in_range=$in_range, match=$(result == in_range)")
    end
    
    # Test with extreme exponents
    println("\nExtreme exponents (should not affect canonicality):")
    x_max_exp = Float32x(0.75f0, typemax(Int32))
    x_min_exp = Float32x(0.75f0, typemin(Int32))
    println("Max exponent: ", is_canonical(x_max_exp))
    println("Min exponent: ", is_canonical(x_min_exp))
end


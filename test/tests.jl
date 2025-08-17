using Test
using .Float32xs

@testset "Float32x Comprehensive Tests" begin
    
    # ==================== Construction and Constants ====================
    @testset "Construction and Constants" begin
        # Basic construction
        @test Float32x(1.5f0, Int32(0)).significand == 1.5f0
        @test Float32x(1.5f0, Int32(0)).exponent == Int32(0)
        
        # Normalization
        x = Float32x(3.0f0, Int32(0))
        @test 1.0f0 <= x.significand < 2.0f0
        @test x.exponent == Int32(1)
        
        x = Float32x(0.5f0, Int32(0))
        @test 1.0f0 <= x.significand < 2.0f0
        @test x.exponent == Int32(-1)
        
        # Special values
        @test isnan(Float32x(NaN32, Int32(0)))
        @test isinf(Float32x(Inf32, Int32(0)))
        @test isinf(Float32x(-Inf32, Int32(0)))
        @test iszero(Float32x(0.0f0, Int32(0)))
        @test iszero(Float32x(-0.0f0, Int32(0)))
        
        # Constants
        @test iszero(zero(Float32x))
        @test isone(one(Float32x))
        @test isinf(inf(Float32x))
        @test isnan(nan(Float32x))
        @test eps(Float32x) == Float32x(eps(Float32))
        @test isinf(typemax(Float32x)) && !signbit(typemax(Float32x))
        @test isinf(typemin(Float32x)) && signbit(typemin(Float32x))
        
        # New: floatmin and floatmax
        @test floatmin(Float32x).significand == 1.0f0
        @test floatmin(Float32x).exponent == typemin(Int32)
        @test floatmax(Float32x).significand == prevfloat(2.0f0)
        @test floatmax(Float32x).exponent == typemax(Int32)
        @test floatmin(Float32x) > zero(Float32x)
        @test floatmax(Float32x) < inf(Float32x)
    end
    
    # ==================== Predicates ====================
    @testset "Predicates" begin
        @test iszero(Float32x(0.0f0, Int32(0)))
        @test iszero(Float32x(-0.0f0, Int32(0)))
        @test !iszero(Float32x(1.0f0, Int32(0)))
        
        @test isone(Float32x(1.0f0, Int32(0)))
        @test !isone(Float32x(2.0f0, Int32(0)))
        @test !isone(Float32x(1.0f0, Int32(1)))
        
        @test isinf(Float32x(Inf32, Int32(0)))
        @test isinf(Float32x(-Inf32, Int32(0)))
        @test !isinf(Float32x(1.0f0, Int32(0)))
        
        @test isnan(Float32x(NaN32, Int32(0)))
        @test !isnan(Float32x(1.0f0, Int32(0)))
        
        @test isfinite(Float32x(1.0f0, Int32(0)))
        @test !isfinite(Float32x(Inf32, Int32(0)))
        @test !isfinite(Float32x(NaN32, Int32(0)))
        
        @test !signbit(Float32x(1.0f0, Int32(0)))
        @test signbit(Float32x(-1.0f0, Int32(0)))
        @test !signbit(Float32x(0.0f0, Int32(0)))
        @test signbit(Float32x(-0.0f0, Int32(0)))
    end
    
    # ==================== Comparisons ====================
    @testset "Comparison Operations" begin
        a = Float32x(1.0f0, Int32(0))
        b = Float32x(2.0f0, Int32(0))
        c = Float32x(1.0f0, Int32(1))  # 1.0 * 2^1 = 2.0
        
        # Equality
        @test a == a
        @test b == c
        @test !(a == b)
        @test Float32x(0.0f0, Int32(0)) == Float32x(-0.0f0, Int32(0))  # -0 == 0
        @test !(Float32x(NaN32, Int32(0)) == Float32x(NaN32, Int32(0)))  # NaN != NaN
        
        # Inequality
        @test a != b
        @test !(a != a)
        @test !(b != c)
        
        # Less than
        @test a < b
        @test a < c
        @test !(b < a)
        @test !(b < c)
        @test Float32x(-1.0f0, Int32(0)) < Float32x(1.0f0, Int32(0))
        @test Float32x(-Inf32, Int32(0)) < Float32x(-1.0f0, Int32(0))
        @test Float32x(1.0f0, Int32(0)) < Float32x(Inf32, Int32(0))
        
        # Special case: NaN comparisons
        nan_val = Float32x(NaN32, Int32(0))
        @test !(nan_val < a)
        @test !(a < nan_val)
        @test !(nan_val < nan_val)
        
        # Less than or equal
        @test a <= a
        @test a <= b
        @test !(b <= a)
        
        # Greater than
        @test b > a
        @test !(a > b)
        @test !(a > a)
        
        # Greater than or equal
        @test b >= a
        @test b >= b
        @test !(a >= b)
        
        # isequal (distinguishes -0.0 from 0.0 and NaN equals itself)
        @test isequal(Float32x(NaN32, Int32(0)), Float32x(NaN32, Int32(0)))
        @test !isequal(Float32x(0.0f0, Int32(0)), Float32x(-0.0f0, Int32(0)))
    end
    
    # ==================== Conversions ====================
    @testset "Conversions" begin
        # To Float32x
        @test Float32x(5) == Float32x(5.0f0, Int32(0))
        @test Float32x(5.0f0) == Float32x(5.0f0, Int32(0))
        @test isnan(Float32x(NaN))
        @test isinf(Float32x(Inf)) && !signbit(Float32x(Inf))
        @test isinf(Float32x(-Inf)) && signbit(Float32x(-Inf))
        
        # From Float32x to Float64
        x = Float32x(1.5f0, Int32(10))
        @test Float64(x) ≈ 1.5 * 2^10
        
        # Overflow/underflow in conversion to Float64
        @test isinf(Float64(Float32x(1.0f0, Int32(1100))))  # Overflow
        @test Float64(Float32x(1.0f0, Int32(-1100))) == 0.0  # Underflow
        
        # From Float32x to Float32
        @test Float32(Float32x(1.5f0, Int32(0))) == 1.5f0
        
        # Overflow/underflow in conversion to Float32
        @test isinf(Float32(Float32x(1.0f0, Int32(200))))  # Overflow
        @test Float32(Float32x(1.0f0, Int32(-200))) == 0.0f0  # Underflow
        
        # Round-trip conversions
        for val in [1.0, -2.5, 0.0, -0.0]
            @test Float64(Float32x(val)) ≈ val
        end
        
        # Special values
        @test isnan(Float64(Float32x(NaN32, Int32(0))))
        @test isinf(Float64(Float32x(Inf32, Int32(0)))) && Float64(Float32x(Inf32, Int32(0))) > 0
        @test isinf(Float64(Float32x(-Inf32, Int32(0)))) && Float64(Float32x(-Inf32, Int32(0))) < 0
        
        # Signed zero preservation
        @test signbit(Float64(Float32x(-0.0f0, Int32(0))))
        @test !signbit(Float64(Float32x(0.0f0, Int32(0))))
        @test signbit(Float32(Float32x(-0.0f0, Int32(0))))
        @test !signbit(Float32(Float32x(0.0f0, Int32(0))))
        
        # New: BigFloat conversions
        @testset "BigFloat Conversions" begin
            # To BigFloat
            a = Float32x(1.5f0, Int32(10))
            big_a = BigFloat(a)
            @test big_a == BigFloat(1.5) * BigFloat(2)^10
            
            # From BigFloat
            big_b = BigFloat(π)
            b = Float32x(big_b)
            @test abs(Float64(b) - Float64(big_b)) < 1e-6  # Lost precision is expected
            
            # Special values
            @test isnan(BigFloat(nan(Float32x)))
            @test isinf(BigFloat(inf(Float32x))) && BigFloat(inf(Float32x)) > 0
            @test isinf(BigFloat(Float32x(-Inf32, Int32(0)))) && BigFloat(Float32x(-Inf32, Int32(0))) < 0
            @test iszero(BigFloat(zero(Float32x)))
            @test signbit(BigFloat(Float32x(-0.0f0, Int32(0))))
            
            # Large exponents
            huge = Float32x(1.5f0, Int32(1000))
            big_huge = BigFloat(huge)
            @test big_huge == BigFloat(1.5) * BigFloat(2)^1000
            
            # From BigFloat with overflow
            big_overflow = BigFloat(2)^(BigFloat(typemax(Int32)) * 2)
            @test isinf(Float32x(big_overflow))
            
            # From BigFloat with underflow
            big_underflow = BigFloat(2)^(BigFloat(typemin(Int32)) * 2)
            @test iszero(Float32x(big_underflow))
            
            # Round-trip for normal values
            for val in [1.5, -2.25, 100.0, -0.125]
                x = Float32x(val)
                @test Float64(Float32x(BigFloat(x))) ≈ Float64(x)
            end
            
            # Promotion rule
            @test promote_type(Float32x, BigFloat) == BigFloat
            x = Float32x(2.0f0, Int32(0))
            y = BigFloat(3)
            @test x + y isa BigFloat
            @test x * y isa BigFloat
        end
    end
    
    # ==================== Arithmetic Operations ====================
    @testset "Addition" begin
        a = Float32x(1.0f0, Int32(0))
        b = Float32x(2.0f0, Int32(0))
        
        @test a + b == Float32x(3.0f0, Int32(0))
        @test a + zero(Float32x) == a
        @test zero(Float32x) + a == a
        
        # Different exponents
        c = Float32x(1.0f0, Int32(10))
        d = Float32x(1.0f0, Int32(5))
        result = c + d
        @test Float64(result) ≈ 2^10 + 2^5
        
        # Large exponent difference (should return larger)
        e = Float32x(1.0f0, Int32(0))
        f = Float32x(1.0f0, Int32(30))
        @test f + e == f  # e is negligible
        
        # Special values
        @test isnan(a + nan(Float32x))
        @test isinf(a + inf(Float32x))
        @test isnan(inf(Float32x) + Float32x(-Inf32, Int32(0)))  # Inf + (-Inf) = NaN
        
        # Signed zero
        @test Float32x(0.0f0, Int32(0)) + Float32x(-0.0f0, Int32(0)) == Float32x(0.0f0, Int32(0))
    end
    
    @testset "Subtraction" begin
        a = Float32x(3.0f0, Int32(0))
        b = Float32x(2.0f0, Int32(0))
        
        @test a - b == Float32x(1.0f0, Int32(0))
        @test a - a == zero(Float32x)
        @test a - zero(Float32x) == a
        
        # Negation
        @test -a == Float32x(-3.0f0, Int32(0))
        @test -zero(Float32x) == zero(Float32x)
        @test signbit(-Float32x(0.0f0, Int32(0)))
        @test !signbit(-Float32x(-0.0f0, Int32(0)))
    end
    
    @testset "Multiplication" begin
        a = Float32x(2.0f0, Int32(0))
        b = Float32x(3.0f0, Int32(0))
        
        @test a * b == Float32x(6.0f0, Int32(0))
        @test a * one(Float32x) == a
        @test a * zero(Float32x) == zero(Float32x)
        
        # Different exponents
        c = Float32x(1.5f0, Int32(10))
        d = Float32x(2.0f0, Int32(5))
        result = c * d
        @test Float64(result) ≈ 1.5 * 2^10 * 2.0 * 2^5
        
        # Special values
        @test isnan(zero(Float32x) * inf(Float32x))  # 0 * Inf = NaN
        @test isnan(inf(Float32x) * zero(Float32x))  # Inf * 0 = NaN
        @test isinf(Float32x(2.0f0, Int32(0)) * inf(Float32x))
        
        # Sign handling
        @test signbit(Float32x(-1.0f0, Int32(0)) * Float32x(1.0f0, Int32(0)))
        @test !signbit(Float32x(-1.0f0, Int32(0)) * Float32x(-1.0f0, Int32(0)))
        
        # Exponent overflow
        huge = Float32x(1.0f0, typemax(Int32) ÷ 2)
        @test isinf(huge * huge)
    end
    
    @testset "Division" begin
        a = Float32x(6.0f0, Int32(0))
        b = Float32x(2.0f0, Int32(0))
        
        @test a / b == Float32x(3.0f0, Int32(0))
        @test a / one(Float32x) == a
        
        # Division by zero
        @test isinf(a / zero(Float32x))
        @test isnan(zero(Float32x) / zero(Float32x))  # 0/0 = NaN
        
        # Special values
        @test isnan(inf(Float32x) / inf(Float32x))  # Inf/Inf = NaN
        @test iszero(a / inf(Float32x))  # finite/Inf = 0
        @test isinf(inf(Float32x) / a)  # Inf/finite = Inf
        
        # Sign handling
        @test signbit(Float32x(-6.0f0, Int32(0)) / Float32x(2.0f0, Int32(0)))
        @test !signbit(Float32x(-6.0f0, Int32(0)) / Float32x(-2.0f0, Int32(0)))
        
        # Exponent underflow
        tiny = Float32x(1.0f0, typemin(Int32) ÷ 2)
        huge = Float32x(1.0f0, typemax(Int32) ÷ 2)
        @test iszero(tiny / huge)
    end
    
    @testset "Power" begin
        a = Float32x(2.0f0, Int32(0))
        
        # Integer powers
        @test a^0 == one(Float32x)
        @test a^1 == a
        @test a^2 == Float32x(4.0f0, Int32(0))
        @test a^3 == Float32x(8.0f0, Int32(0))
        @test a^(-1) == Float32x(0.5f0, Int32(0))
        @test a^(-2) == Float32x(0.25f0, Int32(0))
        
        # Special cases
        @test isone(one(Float32x)^inf(Float32x))  # 1^Inf = 1
        @test isone(Float32x(5.0f0, Int32(0))^zero(Float32x))  # x^0 = 1
        @test iszero(zero(Float32x)^Float32x(2.0f0, Int32(0)))  # 0^positive = 0
        @test isinf(zero(Float32x)^Float32x(-2.0f0, Int32(0)))  # 0^negative = Inf
        
        # Negative base with non-integer exponent
        @test isnan(Float32x(-2.0f0, Int32(0))^Float32x(1.5f0, Int32(0)))
    end
    
    # ==================== Pre-arithmetic Operations ====================
    @testset "Pre-arithmetic Operations" begin
        a = Float32x(-3.5f0, Int32(2))
        
        @test abs(a) == Float32x(3.5f0, Int32(2))
        @test abs(Float32x(-0.0f0, Int32(0))) == Float32x(0.0f0, Int32(0))
        
        @test sign(a) == Float32x(-1.0f0, Int32(0))
        @test sign(Float32x(5.0f0, Int32(0))) == Float32x(1.0f0, Int32(0))
        @test sign(zero(Float32x)) == zero(Float32x)
        @test signbit(sign(Float32x(-0.0f0, Int32(0))))  # sign preserves signed zero
        
        b = Float32x(2.0f0, Int32(0))
        @test copysign(b, a) == Float32x(-2.0f0, Int32(0))
        @test copysign(a, b) == Float32x(3.5f0, Int32(2))
        
        @test flipsign(b, a) == Float32x(-2.0f0, Int32(0))
        @test flipsign(b, b) == b
        
        # nextfloat/prevfloat
        c = Float32x(1.0f0, Int32(0))
        next_c = nextfloat(c)
        @test next_c > c
        @test prevfloat(next_c) == c
        
        # Test boundary crossing
        d = Float32x(prevfloat(2.0f0), Int32(0))
        next_d = nextfloat(d)
        @test next_d.significand == 1.0f0
        @test next_d.exponent == Int32(1)
        
        # Special cases
        @test isnan(nextfloat(nan(Float32x)))
        @test nextfloat(inf(Float32x)) == inf(Float32x)
        @test prevfloat(Float32x(-Inf32, Int32(0))) == Float32x(-Inf32, Int32(0))
    end
    
    # ==================== Rounding Operations ====================
    @testset "Rounding Operations" begin
        a = Float32x(3.7f0, Int32(0))
        b = Float32x(3.2f0, Int32(0))
        c = Float32x(-3.7f0, Int32(0))
        
        @test Float64(round(a)) ≈ 4.0
        @test Float64(round(b)) ≈ 3.0
        @test Float64(round(c)) ≈ -4.0
        
        @test Float64(floor(a)) ≈ 3.0
        @test Float64(floor(b)) ≈ 3.0
        @test Float64(floor(c)) ≈ -4.0
        
        @test Float64(ceil(a)) ≈ 4.0
        @test Float64(ceil(b)) ≈ 4.0
        @test Float64(ceil(c)) ≈ -3.0
        
        @test Float64(trunc(a)) ≈ 3.0
        @test Float64(trunc(b)) ≈ 3.0
        @test Float64(trunc(c)) ≈ -3.0
        
        # Special values
        @test isnan(round(nan(Float32x)))
        @test isinf(round(inf(Float32x)))
        
        # Large values (no rounding needed)
        huge = Float32x(1.0f0, Int32(60))
        @test round(huge) == huge
        
        # significand and exponent extraction
        d = Float32x(1.5f0, Int32(10))
        @test significand(d) == Float64(1.5f0)
        @test exponent(d) == Int32(10)
    end
    
    # ==================== Mathematical Functions ====================
    @testset "Exponential Functions" begin
        a = Float32x(2.0f0, Int32(0))
        
        @test Float64(exp(zero(Float32x))) ≈ 1.0
        @test Float64(exp(one(Float32x))) ≈ ℯ
        @test Float64(exp2(a)) ≈ 4.0
        @test Float64(exp10(one(Float32x))) ≈ 10.0
        @test Float64(expm1(zero(Float32x))) ≈ 0.0
        
        # Special values
        @test isinf(exp(inf(Float32x)))
        @test iszero(exp(Float32x(-Inf32, Int32(0))))
        @test isnan(exp(nan(Float32x)))
        
        # Overflow/underflow
        @test isinf(exp(Float32x(100.0f0, Int32(0))))  # Overflow
        @test iszero(exp(Float32x(-100.0f0, Int32(0))))  # Underflow
    end
    
    @testset "Logarithmic Functions" begin
        a = Float32x(2.0f0, Int32(0))
        e_val = Float32x(Float32(ℯ), Int32(0))
        
        @test Float64(log(one(Float32x))) ≈ 0.0
        @test Float64(log(e_val)) ≈ 1.0 atol=1e-6
        @test Float64(log2(a)) ≈ 1.0
        @test Float64(log10(Float32x(10.0f0, Int32(0)))) ≈ 1.0
        @test Float64(log1p(zero(Float32x))) ≈ 0.0
        
        # Special values
        @test isinf(log(zero(Float32x))) && signbit(log(zero(Float32x)))  # log(0) = -Inf
        @test isnan(log(Float32x(-1.0f0, Int32(0))))  # log(negative) = NaN
        @test isinf(log(inf(Float32x)))  # log(Inf) = Inf
        @test isnan(log(nan(Float32x)))
        
        # log1p special cases
        @test isinf(log1p(Float32x(-1.0f0, Int32(0)))) && signbit(log1p(Float32x(-1.0f0, Int32(0))))
        @test isnan(log1p(Float32x(-2.0f0, Int32(0))))
    end
    
    @testset "Root Functions" begin
        a = Float32x(4.0f0, Int32(0))
        b = Float32x(8.0f0, Int32(0))
        
        @test Float64(sqrt(a)) ≈ 2.0
        @test Float64(sqrt(Float32x(1.0f0, Int32(10)))) ≈ 2^5  # sqrt(2^10) = 2^5
        @test Float64(cbrt(b)) ≈ 2.0
        
        # Special values
        @test iszero(sqrt(zero(Float32x)))
        @test signbit(sqrt(Float32x(-0.0f0, Int32(0))))  # sqrt(-0) = -0
        @test isnan(sqrt(Float32x(-1.0f0, Int32(0))))  # sqrt(negative) = NaN
        @test isinf(sqrt(inf(Float32x)))
        @test isnan(sqrt(nan(Float32x)))
    end
    
    @testset "Trigonometric Functions" begin
        zero_val = zero(Float32x)
        pi_val = Float32x(Float32(π), Int32(0))
        
        @test Float64(sin(zero_val)) ≈ 0.0
        @test Float64(cos(zero_val)) ≈ 1.0
        @test Float64(tan(zero_val)) ≈ 0.0
        
        @test Float64(sin(pi_val)) ≈ 0.0 atol=1e-6
        @test Float64(cos(pi_val)) ≈ -1.0 atol=1e-6
        
        # Inverse trig
        @test Float64(asin(zero_val)) ≈ 0.0
        @test Float64(acos(one(Float32x))) ≈ 0.0
        @test Float64(atan(zero_val)) ≈ 0.0
        @test Float64(atan(inf(Float32x))) ≈ π/2
        @test Float64(atan(Float32x(-Inf32, Int32(0)))) ≈ -π/2
        
        # Domain errors
        @test isnan(asin(Float32x(2.0f0, Int32(0))))  # asin(2) = NaN
        @test isnan(acos(Float32x(2.0f0, Int32(0))))  # acos(2) = NaN
        
        # Special values
        @test isnan(sin(inf(Float32x)))
        @test isnan(cos(inf(Float32x)))
        @test isnan(tan(inf(Float32x)))
    end
    
    @testset "Hyperbolic Functions" begin
        zero_val = zero(Float32x)
        one_val = one(Float32x)
        
        @test Float64(sinh(zero_val)) ≈ 0.0
        @test Float64(cosh(zero_val)) ≈ 1.0
        @test Float64(tanh(zero_val)) ≈ 0.0
        
        # Inverse hyperbolic
        @test Float64(asinh(zero_val)) ≈ 0.0
        @test Float64(acosh(one_val)) ≈ 0.0
        @test Float64(atanh(zero_val)) ≈ 0.0
        
        # Special values
        @test isinf(sinh(inf(Float32x)))
        @test isinf(cosh(inf(Float32x))) && !signbit(cosh(inf(Float32x)))
        @test Float64(tanh(inf(Float32x))) ≈ 1.0
        @test Float64(tanh(Float32x(-Inf32, Int32(0)))) ≈ -1.0
        
        # Domain errors
        @test isnan(acosh(Float32x(0.5f0, Int32(0))))  # acosh(x<1) = NaN
        @test isinf(atanh(one_val))  # atanh(1) = Inf
        @test isinf(atanh(-one_val)) && signbit(atanh(-one_val))  # atanh(-1) = -Inf
        @test isnan(atanh(Float32x(2.0f0, Int32(0))))  # atanh(|x|>1) = NaN
    end
    
    @testset "Utility Functions" begin
        a = Float32x(1.5f0, Int32(10))
        
        # ldexp
        @test ldexp(a, 5) == Float32x(1.5f0, Int32(15))
        @test ldexp(a, -5) == Float32x(1.5f0, Int32(5))
        
        # Overflow/underflow in ldexp
        @test isinf(ldexp(a, typemax(Int32)))
        @test iszero(ldexp(a, typemin(Int32)))
        
        # New: ldexp with separate significand and exponent
        @test ldexp(1.5, 10, Float32x) == Float32x(1.5f0, Int32(10))
        @test ldexp(3.0, 5, Float32x) == Float32x(1.5f0, Int32(6))  # 3.0 = 1.5 * 2^1
        @test isnan(ldexp(NaN, 0, Float32x))
        @test isinf(ldexp(Inf, 0, Float32x))
        @test iszero(ldexp(0.0, 100, Float32x))
        
        # frexp - enhanced testing
        frac, exp = frexp(a)
        @test frac ≈ 0.75  # 1.5/2
        @test exp == 11  # exponent + 1
        
        # Verify frexp reconstruction: x = frac * 2^exp
        @test Float64(a) ≈ frac * 2^exp
        
        # frexp special cases
        frac_nan, exp_nan = frexp(nan(Float32x))
        @test isnan(frac_nan) && exp_nan == 0
        
        frac_inf, exp_inf = frexp(inf(Float32x))
        @test isinf(frac_inf) && exp_inf == 0
        
        frac_zero, exp_zero = frexp(zero(Float32x))
        @test iszero(frac_zero) && exp_zero == 0
        
        frac_negzero, exp_negzero = frexp(Float32x(-0.0f0, Int32(0)))
        @test iszero(frac_negzero) && signbit(frac_negzero) && exp_negzero == 0
        
        # Test frexp/ldexp round-trip
        for val in [Float32x(1.5f0, Int32(10)), Float32x(-2.5f0, Int32(-5)), 
                    Float32x(1.0f0, Int32(100)), Float32x(1.99f0, Int32(-100))]
            f, e = frexp(val)
            reconstructed = ldexp(f, e, Float32x)
            @test Float64(reconstructed) ≈ Float64(val)
        end
        
        # modf
        b = Float32x(3.7f0, Int32(0))
        fpart, ipart = modf(b)
        @test Float64(ipart) ≈ 3.0
        @test Float64(fpart) ≈ 0.7 atol=1e-6
        
        # Special values in modf
        fpart_inf, ipart_inf = modf(inf(Float32x))
        @test iszero(fpart_inf) && isinf(ipart_inf)
        
        fpart_nan, ipart_nan = modf(nan(Float32x))
        @test iszero(fpart_nan) && isnan(ipart_nan)
        
        # New: Test significand and exponent functions
        @testset "significand and exponent" begin
            x = Float32x(1.5f0, Int32(10))
            
            # significand function
            @test significand(x) == 1.5
            @test isnan(significand(nan(Float32x)))
            @test isinf(significand(inf(Float32x))) && significand(inf(Float32x)) > 0
            @test isinf(significand(Float32x(-Inf32, Int32(0)))) && significand(Float32x(-Inf32, Int32(0))) < 0
            @test iszero(significand(zero(Float32x)))
            @test signbit(significand(Float32x(-0.0f0, Int32(0))))
            
            # exponent function
            @test exponent(x) == Int32(10)
            @test_throws DomainError exponent(nan(Float32x))
            @test_throws DomainError exponent(inf(Float32x))
            @test_throws DomainError exponent(zero(Float32x))
            
            # Test with various values
            y = Float32x(1.0f0, Int32(0))
            @test significand(y) == 1.0
            @test exponent(y) == Int32(0)
            
            z = Float32x(1.99f0, Int32(-50))
            @test significand(z) ≈ 1.99
            @test exponent(z) == Int32(-50)
        end
        
        # New: Test floatmin and floatmax instance methods
        @testset "floatmin and floatmax instance methods" begin
            x = Float32x(1.5f0, Int32(10))
            
            @test floatmin(x) == floatmin(Float32x)
            @test floatmax(x) == floatmax(Float32x)
            
            # These should work for any value
            @test floatmin(nan(Float32x)) == floatmin(Float32x)
            @test floatmax(inf(Float32x)) == floatmax(Float32x)
        end
    end
    
    # ==================== Display and Hashing ====================
    @testset "Display and Hashing" begin
        # Display
        @test string(Float32x(1.5f0, Int32(0))) != ""
        @test string(nan(Float32x)) == "NaN"
        @test string(inf(Float32x)) == "Inf"
        @test string(Float32x(-Inf32, Int32(0))) == "-Inf"
        @test string(zero(Float32x)) == "0.0"
        @test string(Float32x(-0.0f0, Int32(0))) == "-0.0"
        
        # Very large exponents should show in component form
        huge_exp = Float32x(1.5f0, Int32(10000))
        str = string(huge_exp)
        @test occursin("2^", str)
        
        # Hashing
        a = Float32x(1.5f0, Int32(10))
        b = Float32x(1.5f0, Int32(10))
        c = Float32x(1.5f0, Int32(11))
        
        @test hash(a) == hash(b)
        @test hash(a) != hash(c)  # Different values should (usually) have different hashes
        
        # Special values should hash consistently
        @test hash(nan(Float32x)) == hash(nan(Float32x))
        @test hash(inf(Float32x)) == hash(inf(Float32x))
        @test hash(zero(Float32x)) == hash(zero(Float32x))
    end
    
    # ==================== Edge Cases and Stress Tests ====================
    @testset "Edge Cases and Stress Tests" begin
        # Very large and very small exponents
        huge = Float32x(1.0f0, typemax(Int32) ÷ 2)
        tiny = Float32x(1.0f0, typemin(Int32) ÷ 2)
        
        @test isinf(huge * huge)  # Overflow
        @test iszero(tiny * tiny)  # Underflow
        @test isinf(huge / tiny)  # Overflow in division
        @test iszero(tiny / huge)  # Underflow in division
        
        # Chain operations
        result = one(Float32x)
        for i in 1:10
            result = result * Float32x(2.0f0, Int32(0))
        end
        @test Float64(result) ≈ 1024.0
        
        # Mixed operations with special values
        @test isnan(inf(Float32x) - inf(Float32x))
        @test isnan(inf(Float32x) / inf(Float32x))
        @test isnan(zero(Float32x) * inf(Float32x))
        
        # Precision preservation
        a = Float32x(1.0f0, Int32(0))
        b = Float32x(eps(Float32), Int32(0))
        c = a + b
        @test c > a  # Should preserve the small addition
        
        # Subnormal-like behavior (though Float32x doesn't have true subnormals)
        small = Float32x(1.0f0, Int32(-1000))
        smaller = Float32x(1.0f0, Int32(-1001))
        @test small / Float32x(2.0f0, Int32(0)) == smaller
    end
    
    # ==================== Promotion and Mixed Arithmetic ====================
    @testset "Promotion and Mixed Arithmetic" begin
        a = Float32x(2.0f0, Int32(0))
        
        # Mixed with integers
        @test a + 3 == Float32x(5.0f0, Int32(0))
        @test 3 + a == Float32x(5.0f0, Int32(0))
        @test a * 3 == Float32x(6.0f0, Int32(0))
        @test 6 / a == Float32x(3.0f0, Int32(0))
        
        # Mixed with Float32
        @test a + 3.0f0 == Float32x(5.0f0, Int32(0))
        @test 3.0f0 * a == Float32x(6.0f0, Int32(0))
        
        # Mixed with Float64
        @test a + 3.0 == Float32x(5.0f0, Int32(0))
        @test 3.0 * a == Float32x(6.0f0, Int32(0))
        
        # Promotion rules
        @test promote_type(Float32x, Int) == Float32x
        @test promote_type(Float32x, Float32) == Float32x
        @test promote_type(Float32x, Float64) == Float32x
    end
    
    # ==================== Compliance with AbstractFloat Interface ====================
    @testset "AbstractFloat Interface Compliance" begin
        # Verify Float32x is a subtype of AbstractFloat
        @test Float32x <: AbstractFloat
        
        # Test that generic functions work
        a = Float32x(1.5f0, Int32(0))
        
        # These should work for any AbstractFloat
        @test isreal(a)
        @test !isinteger(a)
        @test isinteger(Float32x(2.0f0, Int32(0)))
        
        # Test with arrays (AbstractFloat types should work in arrays)
        arr = [Float32x(1.0f0, Int32(0)), Float32x(2.0f0, Int32(0)), Float32x(3.0f0, Int32(0))]
        @test sum(arr) == Float32x(6.0f0, Int32(0))
        @test maximum(arr) == Float32x(3.0f0, Int32(0))
        @test minimum(arr) == Float32x(1.0f0, Int32(0))
    end
    
end

# Run the tests
println("Starting Float32x test suite...")
println("=" ^ 60)


using DrWatson, Test
@quickactivate "LPFC_adaptation"

# Here you include files using `srcdir`
# include(srcdir("file.jl"))
include(srcdir("Pospischil.jl"))

# Run test suite
println("Starting tests")
ti = time()

@testset "LPFC_adaptation tests" begin
    @test 1 == 1
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")

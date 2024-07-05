# Part 2 visualization 
# using Pkg
# Pkg.add("Plots")
using Plots
df = DataFrame(name=["Julia", "Robert", "Bob","Mary"], age=[12,15,45,32])

# Basic line plot:
plot([1,2,3,4,5],[3,6,9,15,16],title="Basic line chart",label="Line")

# Scatter plot:
plot([1,2,3,4,5],[3,6,9,15,16],title="Basic scatter plot",label="Data",seriestype="scatter")

# Bar charts:
plot(df.name,df.age,title="Ages",label=nothing,seriestype="bar")

